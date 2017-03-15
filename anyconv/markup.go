package anyconv

import (
	"errors"
	"fmt"

	"github.com/unixpickle/anydiff"
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
	chain := convmarkup.RealizerChain{convmarkup.MetaRealizer{},
		Realizer(c)}
	instance, _, err := chain.Realize(convmarkup.Dims{}, block)
	if err != nil {
		return nil, errors.New("realize markup block: " + err.Error())
	}
	if layer, ok := instance.(anynet.Layer); ok {
		return layer, nil
	} else {
		return nil, fmt.Errorf("not any anynet.Layer: %T", instance)
	}
}

// Realizer creates a convmarkup.Realizer capable of
// realizing convolutional networks, residual layers, etc.
//
// Realized objects will all implement anynet.Layer.
//
// The Realizer is meant to be used in conjunction with an
// anyconv.MetaRealizer.
func Realizer(c anyvec.Creator) convmarkup.Realizer {
	return &realizer{creator: c}
}

type realizer struct {
	creator anyvec.Creator
}

func (r *realizer) Realize(chain convmarkup.RealizerChain, inDims convmarkup.Dims,
	b convmarkup.Block) (interface{}, error) {
	switch b := b.(type) {
	case *convmarkup.Root:
		return r.net(chain, inDims, b.Children)
	case *convmarkup.Conv:
		return r.conv(inDims, b)
	case *convmarkup.Residual:
		return r.residual(chain, inDims, b)
	case *convmarkup.FC:
		return r.fc(inDims, b)
	case *convmarkup.Activation:
		return r.activation(inDims, b)
	case *convmarkup.Pool:
		return r.pool(inDims, b)
	case *convmarkup.Padding:
		return r.padding(inDims, b)
	case *convmarkup.Resize:
		return r.resize(inDims, b)
	case *convmarkup.Linear:
		return r.linear(b)
	default:
		return nil, convmarkup.ErrUnsupportedBlock
	}
}

func (r *realizer) net(chain convmarkup.RealizerChain, inDims convmarkup.Dims,
	ch []convmarkup.Block) (anynet.Net, error) {
	var res anynet.Net
	for _, b := range ch {
		// Avoiding nested anynet.Net objects.
		if rep, ok := b.(*convmarkup.Repeat); ok {
			for i := 0; i < rep.N; i++ {
				net, err := r.net(chain, inDims, rep.Children)
				if err != nil {
					return nil, err
				}
				res = append(res, net...)
			}
			continue
		}
		obj, _, err := chain.Realize(inDims, b)
		if err != nil {
			return nil, err
		} else if obj != nil {
			if layer, ok := obj.(anynet.Layer); ok {
				res = append(res, layer)
			} else {
				return nil, fmt.Errorf("not an anynet.Layer: %T", obj)
			}
		}
		inDims = b.OutDims()
	}
	return res, nil
}

func (r *realizer) conv(d convmarkup.Dims, b *convmarkup.Conv) (anynet.Layer, error) {
	res := &Conv{
		FilterWidth:  b.FilterWidth,
		FilterHeight: b.FilterHeight,
		FilterCount:  b.FilterCount,
		StrideX:      b.StrideX,
		StrideY:      b.StrideY,
		InputWidth:   d.Width,
		InputHeight:  d.Height,
		InputDepth:   d.Depth,
	}
	res.InitRand(r.creator)
	return res, nil
}

func (r *realizer) residual(chain convmarkup.RealizerChain, d convmarkup.Dims,
	b *convmarkup.Residual) (anynet.Layer, error) {
	resPart, err := r.net(chain, d, b.Residual)
	if err != nil {
		return nil, err
	}
	res := &Residual{Layer: resPart}
	if len(b.Projection) != 0 {
		projPart, err := r.net(chain, d, b.Projection)
		if err != nil {
			return nil, err
		}
		res.Projection = projPart
	}
	return res, nil
}

func (r *realizer) fc(d convmarkup.Dims, b *convmarkup.FC) (anynet.Layer, error) {
	inSize := d.Width * d.Height * d.Depth
	return anynet.NewFC(r.creator, inSize, b.OutCount), nil
}

func (r *realizer) activation(d convmarkup.Dims, b *convmarkup.Activation) (anynet.Layer, error) {
	switch b.Name {
	case "BatchNorm":
		return NewBatchNorm(r.creator, d.Depth), nil
	case "ReLU":
		return anynet.ReLU, nil
	case "Sigmoid":
		return anynet.Sigmoid, nil
	case "Tanh":
		return anynet.Tanh, nil
	case "Softmax":
		return anynet.LogSoftmax, nil
	default:
		return nil, fmt.Errorf("unknown activation: %s", b.Name)
	}
}

func (r *realizer) pool(d convmarkup.Dims, b *convmarkup.Pool) (anynet.Layer, error) {
	switch b.Name {
	case "MeanPool":
		return &MeanPool{
			SpanX:       b.Width,
			SpanY:       b.Height,
			StrideX:     b.StrideX,
			StrideY:     b.StrideY,
			InputWidth:  d.Width,
			InputHeight: d.Height,
			InputDepth:  d.Depth,
		}, nil
	case "MaxPool":
		return &MaxPool{
			SpanX:       b.Width,
			SpanY:       b.Height,
			StrideX:     b.StrideX,
			StrideY:     b.StrideY,
			InputWidth:  d.Width,
			InputHeight: d.Height,
			InputDepth:  d.Depth,
		}, nil
	default:
		return nil, fmt.Errorf("unknown pooling: %s", b.Name)
	}
}

func (r *realizer) padding(d convmarkup.Dims, b *convmarkup.Padding) (anynet.Layer, error) {
	return &Padding{
		InputWidth:    d.Width,
		InputHeight:   d.Height,
		InputDepth:    d.Depth,
		PaddingTop:    b.Top,
		PaddingRight:  b.Right,
		PaddingBottom: b.Bottom,
		PaddingLeft:   b.Left,
	}, nil
}

func (r *realizer) resize(d convmarkup.Dims, b *convmarkup.Resize) (anynet.Layer, error) {
	return &Resize{
		InputWidth:   d.Width,
		InputHeight:  d.Height,
		Depth:        d.Depth,
		OutputWidth:  b.Out.Width,
		OutputHeight: b.Out.Height,
	}, nil
}

func (r *realizer) linear(b *convmarkup.Linear) (anynet.Layer, error) {
	scalerVec := r.creator.MakeVector(1)
	scalerVec.AddScaler(r.creator.MakeNumeric(b.Scale))
	biasVec := r.creator.MakeVector(1)
	biasVec.AddScaler(r.creator.MakeNumeric(b.Bias))
	return &anynet.ParamHider{
		Layer: &anynet.Affine{
			Scalers: anydiff.NewVar(scalerVec),
			Biases:  anydiff.NewVar(biasVec),
		},
	}, nil
}
