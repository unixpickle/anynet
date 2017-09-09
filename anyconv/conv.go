// Package anyconv provides various types of layers for
// convolutional neural networks.
package anyconv

import (
	"errors"
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var c Conv
	serializer.RegisterTypedDeserializer(c.SerializerType(), DeserializeConv)
}

// Conv is a convolutional layer.
//
// All input and output tensors are row-major depth-minor.
type Conv struct {
	FilterCount  int
	FilterWidth  int
	FilterHeight int

	StrideX int
	StrideY int

	InputWidth  int
	InputHeight int
	InputDepth  int

	Filters *anydiff.Var
	Biases  *anydiff.Var

	Conver Conver
}

// DeserializeConv deserialize a Conv.
//
// The Conver is automatically set.
func DeserializeConv(d []byte) (*Conv, error) {
	var inW, inH, inD, fW, fH, sX, sY serializer.Int
	var f, b *anyvecsave.S
	err := serializer.DeserializeAny(d, &inW, &inH, &inD, &fW, &fH, &sX, &sY, &f, &b)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Conv", err)
	}
	res := Conv{
		FilterCount:  f.Vector.Len() / int(fW*fH*inD),
		FilterWidth:  int(fW),
		FilterHeight: int(fH),
		StrideX:      int(sX),
		StrideY:      int(sY),

		InputWidth:  int(inW),
		InputHeight: int(inH),
		InputDepth:  int(inD),

		Filters: anydiff.NewVar(f.Vector),
		Biases:  anydiff.NewVar(b.Vector),
	}
	res.Conver = CurrentConverMaker()(res)
	return &res, nil
}

// InitRand the biases an filters in a randomized fashion
// and sets the Conver.
func (c *Conv) InitRand(cr anyvec.Creator) {
	c.InitZero(cr)

	normalizer := 1 / math.Sqrt(float64(c.FilterWidth*c.FilterHeight*c.InputDepth))
	anyvec.Rand(c.Filters.Vector, anyvec.Normal, nil)
	c.Filters.Vector.Scale(cr.MakeNumeric(normalizer))
}

// InitZero initializes the layer to zero and sets the
// Conver.
func (c *Conv) InitZero(cr anyvec.Creator) {
	filterSize := c.FilterWidth * c.FilterHeight * c.InputDepth
	c.Filters = anydiff.NewVar(cr.MakeVector(filterSize * c.FilterCount))
	c.Biases = anydiff.NewVar(cr.MakeVector(c.FilterCount))
	c.Conver = CurrentConverMaker()(*c)
}

// OutputWidth returns the width of the output tensor.
func (c *Conv) OutputWidth() int {
	w := 1 + (c.InputWidth-c.FilterWidth)/c.StrideX
	if w < 0 {
		return 0
	} else {
		return w
	}
}

// OutputHeight returns the height of the output tensor.
func (c *Conv) OutputHeight() int {
	h := 1 + (c.InputHeight-c.FilterHeight)/c.StrideY
	if h < 0 {
		return 0
	} else {
		return h
	}
}

// OutputDepth returns the depth of the output tensor.
func (c *Conv) OutputDepth() int {
	return c.FilterCount
}

// Apply applies the layer to an input tensor using the
// Conver.
//
// The layer must have been initialized.
func (c *Conv) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	return c.Conver.Apply(in, batchSize)
}

// Parameters returns the layer's parameters.
// The filters come before the biases in the resulting
// slice.
//
// If the layer is uninitialized, the result is nil.
func (c *Conv) Parameters() []*anydiff.Var {
	if c.Filters == nil || c.Biases == nil {
		return nil
	}
	return []*anydiff.Var{c.Filters, c.Biases}
}

// SerializerType returns the unique ID used to serialize
// a Conv with the serializer package.
func (c *Conv) SerializerType() string {
	return "github.com/unixpickle/anynet/anyconv.Conv"
}

// Serialize serializes the layer.
//
// If the layer was not yet initialized, this fails.
func (c *Conv) Serialize() ([]byte, error) {
	if c.Filters == nil || c.Biases == nil {
		return nil, errors.New("cannot serialize uninitialized Conv")
	}
	return serializer.SerializeAny(
		serializer.Int(c.InputWidth),
		serializer.Int(c.InputHeight),
		serializer.Int(c.InputDepth),
		serializer.Int(c.FilterWidth),
		serializer.Int(c.FilterHeight),
		serializer.Int(c.StrideX),
		serializer.Int(c.StrideY),
		&anyvecsave.S{Vector: c.Filters.Vector},
		&anyvecsave.S{Vector: c.Biases.Vector},
	)
}
