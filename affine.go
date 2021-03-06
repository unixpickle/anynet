package anynet

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var a Affine
	serializer.RegisterTypedDeserializer(a.SerializerType(), DeserializeAffine)
	var c ConstAffine
	serializer.RegisterTypedDeserializer(c.SerializerType(), DeserializeConstAffine)
}

// Affine is a layer which performs component-wise affine
// transformations.
//
// In other words, for every component x[i], it computes
//
//     a[i%len(a)]*x[i] + b[i%len(b)]
//
// The scaler vector a and bias vector b are learnable
// variables.
type Affine struct {
	Scalers *anydiff.Var
	Biases  *anydiff.Var
}

// NewAffine creates an Affine layer with one scaler and
// bias.
func NewAffine(c anyvec.Creator, scaler, bias float64) *Affine {
	scalerVec := c.MakeVector(1)
	biasVec := c.MakeVector(1)
	scalerVec.AddScalar(c.MakeNumeric(scaler))
	biasVec.AddScalar(c.MakeNumeric(bias))
	return &Affine{
		Scalers: anydiff.NewVar(scalerVec),
		Biases:  anydiff.NewVar(biasVec),
	}
}

// DeserializeAffine deserializes an Affine layer.
func DeserializeAffine(d []byte) (*Affine, error) {
	var s, b *anyvecsave.S
	if err := serializer.DeserializeAny(d, &s, &b); err != nil {
		return nil, essentials.AddCtx("deserialize Affine", err)
	}
	return &Affine{
		Scalers: anydiff.NewVar(s.Vector),
		Biases:  anydiff.NewVar(b.Vector),
	}, nil
}

// Apply applies the layer to a batch of inputs.
// The size of each vector in the batch must be divisible
// by the number of scalers and biases.
func (a *Affine) Apply(in anydiff.Res, n int) anydiff.Res {
	if in.Output().Len()%n != 0 {
		panic("input size not divisible by batch size")
	}
	inLen := in.Output().Len() / n
	if inLen%a.Scalers.Vector.Len() != 0 || inLen%a.Biases.Vector.Len() != 0 {
		panic("scaler and bias count must divide input count")
	}
	return anydiff.ScaleAddRepeated(in, a.Scalers, a.Biases)
}

// Parameters returns a slice containing the scalers
// followed by the biases.
func (a *Affine) Parameters() []*anydiff.Var {
	return []*anydiff.Var{a.Scalers, a.Biases}
}

// SerializerType returns the unique ID used to serialize
// an Affine with the serializer package.
func (a *Affine) SerializerType() string {
	return "github.com/unixpickle/anynet.Affine"
}

// Serialize serializes the layer.
func (a *Affine) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		&anyvecsave.S{Vector: a.Scalers.Vector},
		&anyvecsave.S{Vector: a.Biases.Vector},
	)
}

// ConstAffine is a layer which performs component-wise
// affine transformations with a constant bias and scaler.
//
// In other words, each component x is transformed via:
//
//     a*x + b
//
type ConstAffine struct {
	Scale float64
	Bias  float64
}

// DeserializeConstAffine deserializes a ConstAffine.
func DeserializeConstAffine(d []byte) (*ConstAffine, error) {
	var res ConstAffine
	if err := serializer.DeserializeAny(d, &res.Scale, &res.Bias); err != nil {
		return nil, essentials.AddCtx("deserialize ConstAffine", err)
	}
	return &res, nil
}

// Apply applies the affine transformation.
func (c *ConstAffine) Apply(in anydiff.Res, n int) anydiff.Res {
	cr := in.Output().Creator()
	return anydiff.AddScalar(
		anydiff.Scale(in, cr.MakeNumeric(c.Scale)),
		cr.MakeNumeric(c.Bias),
	)
}

// SerializerType returns the unique ID used to serialize
// a ConstAffine with the serializer package.
func (c *ConstAffine) SerializerType() string {
	return "github.com/unixpickle/anynet.ConstAffine"
}

// Serialize serializes a ConstAffine.
func (c *ConstAffine) Serialize() ([]byte, error) {
	return serializer.SerializeAny(c.Scale, c.Bias)
}
