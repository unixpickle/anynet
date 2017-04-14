package anynet

import (
	"errors"
	"fmt"
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var f FC
	serializer.RegisterTypedDeserializer(f.SerializerType(), DeserializeFC)
}

// FC is a fully-connected layer.
type FC struct {
	InCount  int
	OutCount int
	Weights  *anydiff.Var
	Biases   *anydiff.Var
}

// DeserializeFC attempts to deserialize an FC.
func DeserializeFC(d []byte) (*FC, error) {
	var weights, biases *anyvecsave.S
	if err := serializer.DeserializeAny(d, &weights, &biases); err != nil {
		return nil, essentials.AddCtx("deserialize FC", err)
	}
	inCount := weights.Vector.Len() / biases.Vector.Len()
	outCount := biases.Vector.Len()
	if inCount*outCount != weights.Vector.Len() {
		return nil, errors.New("deserialize FC: invalid matrix dimensions")
	}
	return &FC{
		InCount:  inCount,
		OutCount: outCount,
		Weights:  anydiff.NewVar(weights.Vector),
		Biases:   anydiff.NewVar(biases.Vector),
	}, nil
}

// NewFC creates a new, randomized FC.
// The randomization scheme targets an output variance of
// 1, given that the input variance is 1.
func NewFC(c anyvec.Creator, in, out int) *FC {
	res := NewFCZero(c, in, out)
	anyvec.Rand(res.Weights.Vector, anyvec.Normal, nil)
	res.Weights.Vector.Scale(c.MakeNumeric(1 / math.Sqrt(float64(in))))
	return res
}

// NewFCZero creates a new, zero'd out FC.
func NewFCZero(c anyvec.Creator, in, out int) *FC {
	return &FC{
		InCount:  in,
		OutCount: out,
		Weights:  anydiff.NewVar(c.MakeVector(in * out)),
		Biases:   anydiff.NewVar(c.MakeVector(out)),
	}
}

// Apply applies the fully-connected layer to a batch of
// inputs.
func (f *FC) Apply(in anydiff.Res, batch int) anydiff.Res {
	if batch*f.InCount != in.Output().Len() {
		panic(fmt.Sprintf("input length should be %d, but got %d",
			batch*f.InCount, in.Output().Len()))
	}
	weightMat := &anydiff.Matrix{
		Data: f.Weights,
		Rows: f.OutCount,
		Cols: f.InCount,
	}
	inMat := &anydiff.Matrix{
		Data: in,
		Rows: batch,
		Cols: f.InCount,
	}
	weighted := anydiff.MatMul(false, true, inMat, weightMat)
	return anydiff.AddRepeated(weighted.Data, f.Biases)
}

// AddBias adds a scaler to the biases.
// It returns f for convenience.
func (f *FC) AddBias(val anyvec.Numeric) *FC {
	f.Biases.Vector.AddScalar(val)
	return f
}

// Parameters returns a slice containing the weights
// and the biases, in that order.
func (f *FC) Parameters() []*anydiff.Var {
	return []*anydiff.Var{f.Weights, f.Biases}
}

// SerializerType returns the unique ID used to serialize
// an FC with the serializer package.
func (f *FC) SerializerType() string {
	return "github.com/unixpickle/anynet.FC"
}

// Serialize serializes the FC.
func (f *FC) Serialize() ([]byte, error) {
	weights := &anyvecsave.S{Vector: f.Weights.Vector}
	biases := &anyvecsave.S{Vector: f.Biases.Vector}
	return serializer.SerializeAny(weights, biases)
}
