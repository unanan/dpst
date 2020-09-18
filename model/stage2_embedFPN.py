import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from model import embednet

# Refer to: https://github.com/ShuLiu1993/PANet/blob/master/lib/modeling/FPN.py
class topdown_lateral_module(nn.Module):
    """Add a top-down lateral module."""
    def __init__(self, dim_in_top, dim_in_lateral):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        # self.conv_lateral = nn.Sequential(
        #     nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0, bias=False),
        #     nn.GroupNorm(net_utils.get_group_gn(self.dim_out), self.dim_out,
        #                  eps=cfg.GROUP_NORM.EPSILON)
        # )
        self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        # if cfg.FPN.USE_GN:
        #     conv = self.conv_lateral[0]
        # else:
        conv = self.conv_lateral

        nn.init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        # Lateral 1x1 conv
        lat = self.conv_lateral(lateral_blob)
        # Top-down 2x upsampling
        # td = F.upsample(top_blob, size=lat.size()[2:], mode='bilinear')
        td = F.interpolate(top_blob, scale_factor=2, mode='nearest')
        # Sum lateral and top-down
        return lat + td

# Refer to: https://github.com/wannabeOG/Mask-RCNN/blob/master/model.py
class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, input:torch.Tensor):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


# Refer to: https://github.com/wannabeOG/Mask-RCNN/blob/master/model.py
class RPN(nn.Module):
    """Builds the model of Region Proposal Network.
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, self.depth*2, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(self.depth*2, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(self.depth*2, 4 * anchors_per_location, kernel_size=1, stride=1)

    def generate_anchors(self, scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    def generate_pyramid_anchors(self, scales, ratios, feature_shapes, feature_strides,anchor_stride):
        """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.
        Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            anchors.append(self.generate_anchors(scales[i], ratios, feature_shapes[i],
                                            feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)

    def apply_box_deltas(self, boxes, deltas):
        """Applies the given deltas to the given boxes.
        boxes: [N, 4] where each row is y1, x1, y2, x2
        deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
        """
        # Convert to y, x, h, w
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= torch.exp(deltas[:, 2])
        width *= torch.exp(deltas[:, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        result = torch.stack([y1, x1, y2, x2], dim=1)
        return result

    def clip_boxes(self, boxes, window):
        """
        boxes: [N, 4] each col is y1, x1, y2, x2
        window: [4] in the form y1, x1, y2, x2
        """
        boxes = torch.stack( \
            [boxes[:, 0].clamp(float(window[0]), float(window[2])),
             boxes[:, 1].clamp(float(window[1]), float(window[3])),
             boxes[:, 2].clamp(float(window[0]), float(window[2])),
             boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
        return boxes

    def proposal_layer(self, inputs, proposal_count=2000, nms_threshold=0.7, anchors=None, rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2])):
        """Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinment detals to anchors.
        Inputs:
            rpn_probs: [batch, anchors, (bg prob, fg prob)]
            rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
        """

        # Currently only supports batchsize 1
        inputs[0] = inputs[0].squeeze(0)
        inputs[1] = inputs[1].squeeze(0)

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, 1]

        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        std_dev = Variable(torch.from_numpy(np.reshape(rpn_bbox_std_dev, [1, 4])).float(), requires_grad=False)
        # std_dev = std_dev.cuda()

        deltas = deltas * std_dev

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(6000, anchors.size()[0])
        scores, order = scores.sort(descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = deltas[order.data, :]  # TODO: Support batch size > 1 ff.
        anchors = anchors[order.data, :]

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = self.apply_box_deltas(anchors, deltas)

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = 512
        window = np.array([0, 0, height, width]).astype(np.float32)
        boxes = self.clip_boxes(boxes, window)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
        keep = keep[:proposal_count]
        boxes = boxes[keep, :]

        # Normalize dimensions to range of 0 to 1.
        norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
        # norm = norm.cuda()
        normalized_boxes = boxes / norm

        # Add back batch dimension
        normalized_boxes = normalized_boxes.unsqueeze(0)

        return normalized_boxes


    def forward(self, x:torch.Tensor):
        # # Channel last
        # x = x.permute(0,2,3,1)

        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        rpn_rois= self.proposal_layer([rpn_probs, rpn_bbox],
                            anchors=self.generate_pyramid_anchors(
                                (16, 32, 64, 128),
                                (0.8,1.0,1.25),
                                np.array(
                                    [[int(math.ceil(512 / stride)),
                                      int(math.ceil(512 / stride))]
                                     for stride in (4, 8, 16, 32)]),
                                (4, 8, 16, 32),
                                1))
        return rpn_rois #[rpn_class_logits, rpn_probs, rpn_bbox]

# Refer to: https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/sequence_modeling.py
class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input:torch.Tensor):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        print(input.shape)
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class embedFPN(nn.Module):
    def __init__(self,backbone:nn.Module):
        super(embedFPN, self).__init__()

        self.backbone  = backbone

        self.attn4     = embednet(d_model=16)
        self.attn3     = embednet(d_model=32)
        self.attn2     = embednet(d_model=64)
        self.attn1     = embednet(d_model=128)

        # self.p5conv = nn.Conv2D(576, 96, kernel_size=3, stride=2, padding=1, bias=False)
        self.topdown43 = topdown_lateral_module(dim_in_top=96, dim_in_lateral=48)
        self.topdown32 = topdown_lateral_module(dim_in_top=96, dim_in_lateral=24)
        self.topdown21 = topdown_lateral_module(dim_in_top=96, dim_in_lateral=16)

        # RPN
        # self.rpn       = RPN(anchors_per_location=3,anchor_stride=1,depth=96)

        # # Sequence Features
        # self.seqencode = nn.Sequential(
        #         BidirectionalLSTM(1024, 256, 256),
        #         BidirectionalLSTM(256, 256, 256))

        self.mask_conv = nn.Sequential(
            nn.Conv2d(96*3, 1, 1)
        )

    # #Refer to: https://github.com/Hanqer/deep-hough-transform/blob/master/model/network.py
    # def upsample_cat(self, p1, p2, p3):
    #     p1 = F.interpolate(p1, size=(128,128), mode='bilinear')
    #     p2 = F.interpolate(p2, size=(128,128), mode='bilinear')
    #     p3 = F.interpolate(p3, size=(128,128), mode='bilinear')
    #     return torch.cat([p1, p2, p3], dim=1)


    def forward(self,x):
        c1,c2,c3,c4 = self.backbone(x)

        p3 = self.topdown43(self.attn4(c4),self.attn3(c3))
        p2 = self.topdown32(p3,self.attn2(c2))
        p1 = self.topdown21(p2,self.attn1(c1))

        # pcat = self.upsample_cat(p1,p2,p3)
        pcat = torch.cat([p1, p2, p3], dim=1)

        return self.mask_conv(pcat)
        # p5 = self.p5conv(c5)
        # P4 = nn.Add(name="fpn_p4add")([
        #     nn.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        #     nn.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        # P3 = nn.Add(name="fpn_p3add")([
        #     nn.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        #     nn.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        # P2 = nn.Add(name="fpn_p2add")([
        #     nn.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        #     nn.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # # Attach 3x3 conv to all P layers to get the final feature maps.
        # P2 = nn.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        # P3 = nn.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        # P4 = nn.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        # P5 = nn.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)


if __name__ =="__main__":
    # net = topdown_lateral_module(dim_in_top=96, dim_in_lateral=48)
    # x1 = torch.randn(2, 96, 16, 16)
    # x2 = torch.randn(2, 48, 32, 32)
    #
    # output = net(x1,x2)
    # print(output.shape) #2x96x32x32

    from model.mobilenet import MobileNetV3_Small
    net = embedFPN(backbone=MobileNetV3_Small())

    x = torch.randn(2, 3, 512, 512)
    mask = net(x)
    print(mask.shape) #2x1x128x128

    torch.save(net.state_dict(), "test.pth")  # ~5.4MB
