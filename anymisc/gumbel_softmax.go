package anymisc

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

const gumbelEpsilon = 1e-10

func init() {
	serializer.RegisterTypedDeserializer((&GumbelSoftmax{}).SerializerType(),
		DeserializeGumbelSoftmax)
}

// GumbelSoftmax is an anynet.Layer that implements the
// Gumbel-Softmax distribution.
//
// For more, see https://arxiv.org/abs/1611.01144.
type GumbelSoftmax struct {
	Temperature float64
}

// DeserializeGumbelSoftmax deserializes a GumbelSoftmax.
func DeserializeGumbelSoftmax(d []byte) (*GumbelSoftmax, error) {
	var res GumbelSoftmax
	if err := serializer.DeserializeAny(d, &res.Temperature); err != nil {
		return nil, err
	}
	return &res, nil
}

// Apply applies the Gumbel-Softmax to each vector in the
// batch.
func (g *GumbelSoftmax) Apply(in anydiff.Res, n int) anydiff.Res {
	c := in.Output().Creator()
	gumbel := c.MakeVector(in.Output().Len())
	anyvec.Rand(gumbel, anyvec.Uniform, nil)
	for i := 0; i < 2; i++ {
		gumbel.AddScalar(c.MakeNumeric(gumbelEpsilon))
		anyvec.Log(gumbel)
		gumbel.Scale(c.MakeNumeric(-1))
	}
	smIn := anydiff.Scale(anydiff.Add(anydiff.NewConst(gumbel), in),
		c.MakeNumeric(1/g.Temperature))
	return anydiff.Exp(anydiff.LogSoftmax(smIn, in.Output().Len()/n))
}

// SerializerType returns the unique ID used to serialize
// a GumbelSoftmax with the serializer package.
func (g *GumbelSoftmax) SerializerType() string {
	return "github.com/unixpickle/anynet/anymisc.GumbelSoftmax"
}

// Serialize serializes the layer.
func (g *GumbelSoftmax) Serialize() ([]byte, error) {
	return serializer.SerializeAny(g.Temperature)
}
