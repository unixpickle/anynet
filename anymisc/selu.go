package anymisc

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

const (
	seluDefaultAlpha  = 1.6732632423543772848170429916717
	seluDefaultLambda = 1.0507009873554804934193349852946
)

func init() {
	serializer.RegisterTypedDeserializer((&SELU{}).SerializerType(), DeserializeSELU)
}

// SELU implements the scaled exponential linear unit
// activation function.
//
// If a field is 0, a default is used.
// The default values are meant to induce a fixed point of
// mean 0 and variance 1.
//
// For more on SELU, see https://arxiv.org/abs/1706.02515.
type SELU struct {
	Alpha  float64
	Lambda float64
}

// DeserializeSELU deserializes a SELU instance.
func DeserializeSELU(d []byte) (*SELU, error) {
	var res SELU
	if err := serializer.DeserializeAny(d, &res.Alpha, &res.Lambda); err != nil {
		return nil, essentials.AddCtx("deserialize SELU", err)
	}
	return &res, nil
}

// Apply applies the activation function.
func (s *SELU) Apply(in anydiff.Res, n int) anydiff.Res {
	alpha := s.Alpha
	lambda := s.Lambda
	if alpha == 0 {
		alpha = seluDefaultAlpha
	}
	if lambda == 0 {
		lambda = seluDefaultLambda
	}

	c := in.Output().Creator()
	return anydiff.Pool(in, func(in anydiff.Res) anydiff.Res {
		posPart := anydiff.ClipPos(in)
		negPart := clipNeg(in)
		return anydiff.Scale(
			anydiff.AddScalar(
				anydiff.Add(
					posPart,
					anydiff.Scale(anydiff.Exp(negPart), c.MakeNumeric(alpha)),
				),
				c.MakeNumeric(-alpha),
			),
			c.MakeNumeric(lambda),
		)
	})
}

// SerializerType returns the unique ID used to serialize
// a SELU with the serializer package.
func (s *SELU) SerializerType() string {
	return "github.com/unixpickle/anynet/anymisc.SELU"
}

// Serialize serializes the SELU.
func (s *SELU) Serialize() ([]byte, error) {
	return serializer.SerializeAny(s.Alpha, s.Lambda)
}

func clipNeg(vec anydiff.Res) anydiff.Res {
	c := vec.Output().Creator()
	return anydiff.Scale(
		anydiff.ClipPos(anydiff.Scale(vec, c.MakeNumeric(-1))),
		c.MakeNumeric(-1),
	)
}
