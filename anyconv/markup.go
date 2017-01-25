package anyconv

import (
	"errors"
	"fmt"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/convmarkup"
)

// FromMarkup creates a neural network from a markup
// description.
//
// For details on this format, see:
// https://github.com/unixpickle/convmarkup.
func FromMarkup(c anyvec.Creator, code string) (anynet.Layer, error) {
	parsed, err := convmarkup.Parse(code)
	if err != nil {
		return nil, errors.New("parse markup: " + err.Error())
	}
	block, err := parsed.Block(convmarkup.Dims{}, convmarkup.DefaultCreators())
	if err != nil {
		return nil, errors.New("make markup block: " + err.Error())
	}
	return FromMarkupBlock(c, block)
}

// FromMarkupBlock creates a neural network from a parsed
// markup block.
func FromMarkupBlock(c anyvec.Creator, b convmarkup.Block) (anynet.Layer, error) {
	return fromMarkupBlock(convmarkup.Dims{}, c, b)
}

func fromMarkupBlock(inDims convmarkup.Dims, c anyvec.Creator,
	b convmarkup.Block) (anynet.Layer, error) {
	switch b := b.(type) {
	case *convmarkup.Root:
		return netForChildren(inDims, c, b.Children)
	case *convmarkup.Conv:
		return layerForConvBlock(inDims, c, b)
	case *convmarkup.Residual:
		return layerForResidualBlock(inDims, c, b)
	case *convmarkup.FC:
		return anynet.NewFC(c, inDims.Width*inDims.Height*inDims.Depth, b.OutCount), nil
	case *convmarkup.Activation:
		return layerForActivationBlock(inDims, c, b)
	case *convmarkup.MaxPool:
		return &MaxPool{
			SpanX:       b.Width,
			SpanY:       b.Height,
			InputWidth:  inDims.Width,
			InputHeight: inDims.Height,
			InputDepth:  inDims.Depth,
		}, nil
	case *convmarkup.Padding:
		return &Padding{
			InputWidth:    inDims.Width,
			InputHeight:   inDims.Height,
			InputDepth:    inDims.Depth,
			PaddingTop:    b.Top,
			PaddingRight:  b.Right,
			PaddingBottom: b.Bottom,
			PaddingLeft:   b.Left,
		}, nil
	default:
		return nil, fmt.Errorf("unexpected markup block: %s", b.Type())
	}
}

func netForChildren(inDims convmarkup.Dims, c anyvec.Creator,
	ch []convmarkup.Block) (anynet.Layer, error) {
	var res anynet.Net
	for _, x := range ch {
		if _, ok := x.(*convmarkup.Input); ok {
			inDims = x.OutDims()
			continue
		}
		layer, err := fromMarkupBlock(inDims, c, x)
		if err != nil {
			return nil, err
		}
		res = append(res, layer)
		inDims = x.OutDims()
	}
	return res, nil
}

func layerForConvBlock(inDims convmarkup.Dims, c anyvec.Creator,
	b *convmarkup.Conv) (anynet.Layer, error) {
	res := &Conv{
		FilterWidth:  b.FilterWidth,
		FilterHeight: b.FilterHeight,
		FilterCount:  b.FilterCount,
		StrideX:      b.StrideX,
		StrideY:      b.StrideY,
		InputWidth:   inDims.Width,
		InputHeight:  inDims.Height,
		InputDepth:   inDims.Depth,
	}
	res.InitRand(c)
	return res, nil
}

func layerForResidualBlock(inDims convmarkup.Dims, c anyvec.Creator,
	b *convmarkup.Residual) (anynet.Layer, error) {
	resPart, err := netForChildren(inDims, c, b.Residual)
	if err != nil {
		return nil, err
	}
	res := &Residual{Layer: resPart}

	if len(b.Projection) != 0 {
		projPart, err := netForChildren(inDims, c, b.Projection)
		if err != nil {
			return nil, err
		}
		res.Projection = projPart
	}

	return res, nil
}

func layerForActivationBlock(inDims convmarkup.Dims, c anyvec.Creator,
	b *convmarkup.Activation) (anynet.Layer, error) {
	switch b.Name {
	case "BatchNorm":
		return NewBatchNorm(c, inDims.Depth), nil
	case "ReLU":
		return anynet.ReLU, nil
	case "Softmax":
		return anynet.LogSoftmax, nil
	default:
		return nil, fmt.Errorf("unknown activation: %s", b.Name)
	}
}
