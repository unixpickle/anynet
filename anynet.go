// Package anynet provides APIs for running and training
// artificial neural networks.
// It includes sub-packages for common neural network
// variations.
package anynet

import (
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var n Net
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNet)
}

// A Parameterizer is anything with learnable variables.
//
// The parameters of a Parameterizer must be in the same
// order every time Parameters() is called.
type Parameterizer interface {
	Parameters() []*anydiff.Var
}

// A Layer is a composable computation unit for use in a
// neural network.
// In a feed-forward network, each layer's output is fed
// into the next layers input.
//
// A Layer's Apply method is inherently batched.
// The input's length must be divisible by the batch size,
// since the batch size indicates how many equally-long
// vectors are packed into the input vector.
type Layer interface {
	Apply(in anydiff.Res, batchSize int) anydiff.Res
}

// A Net evaluates a list of layers, one after another.
type Net []Layer

// DeserializeNet attempts to deserialize the network.
func DeserializeNet(d []byte) (Net, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Net", err)
	}
	res := make(Net, len(slice))
	for i, x := range slice {
		if layer, ok := x.(Layer); ok {
			res[i] = layer
		} else {
			return nil, fmt.Errorf("deserialize Net: not a Layer: %T", x)
		}
	}
	return res, nil
}

// Apply applies the network to a batch.
// If the network contains no layers, the input is
// returned as output.
func (n Net) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	for _, l := range n {
		in = l.Apply(in, batchSize)
	}
	return in
}

// Parameters returns the parameters of the network.
//
// Every layer which implements Parameterizer will have
// its parameters added to the slice.
// Parameters are ordered from the first layer onwards.
func (n Net) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, x := range n {
		if p, ok := x.(Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Net with the serializer package.
func (n Net) SerializerType() string {
	return "github.com/unixpickle/anynet.Net"
}

// Serialize attempts to serialize the network.
// If any Layer is not a serializer.Serializer,
// this fails.
func (n Net) Serialize() ([]byte, error) {
	var slice []serializer.Serializer
	for _, x := range n {
		if s, ok := x.(serializer.Serializer); ok {
			slice = append(slice, s)
		} else {
			return nil, fmt.Errorf("not a Serializer: %T", x)
		}
	}
	return serializer.SerializeSlice(slice)
}
