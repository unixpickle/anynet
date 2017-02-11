package anynet

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var d Dropout
	serializer.RegisterTypedDeserializer(d.SerializerType(), DeserializeDropout)
}

// A Dropout layer applies dropout regularization.
// When disabled, a dropout layer scales its input to
// compute the "expected output".
type Dropout struct {
	Enabled bool

	// The probability of keeping any given input.
	KeepProb float64
}

// DeserializeDropout deserializes a Dropout.
func DeserializeDropout(d []byte) (*Dropout, error) {
	var enabled serializer.Int
	var keepProb serializer.Float64
	if err := serializer.DeserializeAny(d, &enabled, &keepProb); err != nil {
		return nil, essentials.AddCtx("deserialize Dropout", err)
	}
	return &Dropout{
		Enabled:  enabled == 1,
		KeepProb: float64(keepProb),
	}, nil
}

// Apply applies the layer.
func (d *Dropout) Apply(in anydiff.Res, n int) anydiff.Res {
	c := in.Output().Creator()
	if !d.Enabled {
		return anydiff.Scale(in, c.MakeNumeric(d.KeepProb))
	}
	mask := c.MakeVector(in.Output().Len())
	anyvec.Rand(mask, anyvec.Uniform, nil)
	anyvec.LessThan(mask, c.MakeNumeric(d.KeepProb))
	return anydiff.Mul(in, anydiff.NewConst(mask))
}

// SerializerType returns the unique ID used to serialize
// a Dropout with the serializer package.
func (d *Dropout) SerializerType() string {
	return "github.com/unixpickle/anynet.Dropout"
}

// Serialize serializes the Dropout.
func (d *Dropout) Serialize() ([]byte, error) {
	enabledFlag := serializer.Int(0)
	if d.Enabled {
		enabledFlag = 1
	}
	return serializer.SerializeAny(enabledFlag, serializer.Float64(d.KeepProb))
}
