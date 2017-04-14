package anyconv

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

const defaultBNStabilizer = 1e-3

func init() {
	var b BatchNorm
	serializer.RegisterTypedDeserializer(b.SerializerType(), DeserializeBatchNorm)
}

// BatchNorm is a batch normalization layer.
//
// After a network has finished training, BatchNorm layers
// will typically be replaced with anynet.Affine layers so
// that the normalizations are less noisy.
type BatchNorm struct {
	// InputCount indicates how many components to noramlize.
	//
	// For use after a fully-connected layer, this should be
	// the total number of output neurons.
	// For use after a convolutional layer, this sholud be
	// the number of filters.
	InputCount int

	// Post-normalization affine transform.
	Scalers *anydiff.Var
	Biases  *anydiff.Var

	// Stabilizer prevents numerical instability by adding a
	// small constant to variances to keep them from being 0.
	//
	// If it is 0, a default is used.
	Stabilizer float64
}

// DeserializeBatchNorm deserializes a BatchNorm.
func DeserializeBatchNorm(d []byte) (*BatchNorm, error) {
	var s, b *anyvecsave.S
	var stab serializer.Float64
	if err := serializer.DeserializeAny(d, &s, &b, &stab); err != nil {
		return nil, essentials.AddCtx("deserialize BatchNorm", err)
	}
	return &BatchNorm{
		InputCount: s.Vector.Len(),
		Scalers:    anydiff.NewVar(s.Vector),
		Biases:     anydiff.NewVar(b.Vector),
		Stabilizer: float64(stab),
	}, nil
}

// NewBatchNorm creates a BatchNorm with an input size.
func NewBatchNorm(c anyvec.Creator, inCount int) *BatchNorm {
	oneScaler := c.MakeVector(inCount)
	oneScaler.AddScalar(c.MakeNumeric(1))
	return &BatchNorm{
		InputCount: inCount,
		Scalers:    anydiff.NewVar(oneScaler),
		Biases:     anydiff.NewVar(c.MakeVector(inCount)),
	}
}

// Apply applies the layer to some inputs.
func (b *BatchNorm) Apply(in anydiff.Res, batch int) anydiff.Res {
	if in.Output().Len()%b.InputCount != 0 {
		panic("invalid input size")
	}
	return anydiff.Pool(in, func(in anydiff.Res) anydiff.Res {
		c := in.Output().Creator()

		negMean := negMeanRows(in, b.InputCount)
		secondMoment := meanSquare(in, b.InputCount)
		variance := anydiff.Sub(secondMoment, anydiff.Square(negMean))

		variance = anydiff.AddScalar(variance, c.MakeNumeric(b.stabilizer()))
		normalizer := anydiff.Pow(variance, c.MakeNumeric(-0.5))

		totalScaler := anydiff.Mul(b.Scalers, normalizer)
		return anydiff.Pool(totalScaler, func(totalScaler anydiff.Res) anydiff.Res {
			return anydiff.ScaleAddRepeated(
				in,
				totalScaler,
				anydiff.Add(b.Biases, anydiff.Mul(negMean, totalScaler)),
			)
		})
	})
}

// Parameters returns a slice containing the scales and
// biases, in that order.
func (b *BatchNorm) Parameters() []*anydiff.Var {
	return []*anydiff.Var{b.Scalers, b.Biases}
}

// SerializerType returns the unique ID used to serialize
// a BatchNorm with the serializer package.
func (b *BatchNorm) SerializerType() string {
	return "github.com/unixpickle/anynet/anyconv.BatchNorm"
}

// Serialize serializes the layer.
func (b *BatchNorm) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		&anyvecsave.S{Vector: b.Scalers.Vector},
		&anyvecsave.S{Vector: b.Biases.Vector},
		serializer.Float64(b.Stabilizer),
	)
}

func (b *BatchNorm) stabilizer() float64 {
	if b.Stabilizer == 0 {
		return defaultBNStabilizer
	} else {
		return b.Stabilizer
	}
}
