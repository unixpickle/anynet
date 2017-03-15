package anyrnn

import (
	"errors"
	"fmt"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/convmarkup"
)

// Realizer produces a convmarkup.Realizer for recurrent
// neural networks.
//
// Feed-forward layers are realized by layerChain and then
// wrapped in a LayerBlock.
// It is suggested that you use a layerChain with a
// realizer from anyconv.
//
// On top of the layers supported by layerChain, the
// Realizer adds support for RNN-specific blocks.
// These blocks are LSTM and Vanilla, each of which have
// one required attribute, "out", which specifies the
// block's output size.
//
// The Realizer is meant to be used in the same chain as a
// convmarkup.MetaRealizer.
// For example, you might do:
//
//     convmarkup.RealizerChain{
//         convmarkup.MetaRealizer{},
//         anyrnn.Realizer(creator, convmarkup.RealizerChain{
//             convmarkup.MetaRealizer{},
//             anyconv.Realizer(creator),
//         }),
//     }
//
// Instantiated objects are either a Block or a type
// produced by layerChain.
//
// Recurrent blocks may only be present at the top level
// of a file or within Repeat blocks.
// It is not valid, for example, to use a Vanilla block
// inside a Residual block.
func Realizer(c anyvec.Creator, layerChain convmarkup.RealizerChain) convmarkup.Realizer {
	return &realizer{
		creator:    c,
		layerChain: layerChain,
	}
}

// MarkupCreators returns a map of creators to be used by
// convmarkup when parsing RNN markup files.
// It includes the default creators from convmarkup, in
// addition to creators for custom RNN-specific blocks.
func MarkupCreators() map[string]convmarkup.Creator {
	def := convmarkup.DefaultCreators()
	for _, name := range []string{"LSTM", "Vanilla"} {
		def[name] = markupCreator(name)
	}
	return def
}

type realizer struct {
	creator    anyvec.Creator
	layerChain convmarkup.RealizerChain
}

func (r *realizer) Realize(ch convmarkup.RealizerChain, d convmarkup.Dims,
	b convmarkup.Block) (interface{}, error) {
	switch b := b.(type) {
	case *convmarkup.Root:
		return r.stack(ch, d, b.Children)
	case *convmarkup.Repeat:
		return r.repeat(ch, d, b)
	case *markupBlock:
		return r.block(ch, d, b)
	default:
		return r.layer(ch, d, b)
	}
}

func (r *realizer) stack(ch convmarkup.RealizerChain, d convmarkup.Dims,
	blocks []convmarkup.Block) (Stack, error) {
	var res Stack
	for _, child := range blocks {
		obj, _, err := ch.Realize(d, child)
		if err != nil {
			return nil, err
		}
		d = child.OutDims()
		if obj == nil {
		} else if stack, ok := obj.(Stack); ok {
			// Avoid nesting stacks.
			res = append(res, stack...)
		} else if rnn, ok := obj.(Block); ok {
			res = append(res, rnn)
		} else {
			return nil, fmt.Errorf("not an anyrnn.Block: %T", obj)
		}
	}
	return res, nil
}

func (r *realizer) repeat(ch convmarkup.RealizerChain, d convmarkup.Dims,
	b *convmarkup.Repeat) (interface{}, error) {
	var res Stack
	for i := 0; i < b.N; i++ {
		stack, err := r.stack(ch, d, b.Children)
		if err != nil {
			return nil, err
		}
		res = append(res, stack...)
	}
	return res, nil
}

func (r *realizer) block(ch convmarkup.RealizerChain, d convmarkup.Dims,
	b *markupBlock) (interface{}, error) {
	switch b.Name {
	case "LSTM":
		return NewLSTM(r.creator, d.Volume(), b.Out.Volume()), nil
	case "Vanilla":
		return NewVanilla(r.creator, d.Volume(), b.Out.Volume(), anynet.Tanh), nil
	default:
		panic("unexpected name")
	}
}

func (r *realizer) layer(ch convmarkup.RealizerChain, d convmarkup.Dims,
	b convmarkup.Block) (interface{}, error) {
	obj, supp, err := r.layerChain.Realize(d, b)
	if err == nil {
		if obj == nil {
			return nil, nil
		} else if layer, ok := obj.(anynet.Layer); ok {
			return &LayerBlock{Layer: layer}, nil
		} else {
			return nil, fmt.Errorf("not an anynet.Layer: %T", obj)
		}
	} else if !supp {
		return nil, convmarkup.ErrUnsupportedBlock
	} else {
		return nil, err
	}
}

type markupBlock struct {
	Out  convmarkup.Dims
	Name string
}

func markupCreator(name string) convmarkup.Creator {
	return func(in convmarkup.Dims, attr map[string]float64,
		children []convmarkup.Block) (convmarkup.Block, error) {
		if len(children) > 0 {
			return nil, convmarkup.ErrUnexpectedChildren
		}
		val, ok := attr["out"]
		if !ok {
			return nil, errors.New("missing attribute: out")
		}
		if len(attr) != 1 {
			for name := range attr {
				if name != "out" {
					return nil, errors.New("unexpected attribute: " + name)
				}
			}
		}
		if float64(int(val)) != val || val <= 0 {
			return nil, errors.New("invalid value for out attribute")
		}
		return &markupBlock{
			Out:  convmarkup.Dims{Width: 1, Height: 1, Depth: int(val)},
			Name: name,
		}, nil
	}
}

func (m *markupBlock) Type() string {
	return m.Name
}

func (m *markupBlock) OutDims() convmarkup.Dims {
	return m.Out
}
