package anymisc

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// NewIRNN creates an RNN with ReLU activations and a
// scaled identity state transition matrix.
//
// This is based on https://arxiv.org/abs/1504.00941.
func NewIRNN(c anyvec.Creator, in, out int, scale float64) *anyrnn.Vanilla {
	res := anyrnn.NewVanilla(c, in, out, anynet.ReLU)

	hiddenIndices := make([]int, out)
	for i := range hiddenIndices {
		hiddenIndices[i] = i * (out + 1)
	}
	mapper := c.MakeMapper(out*out, hiddenIndices)
	diagonal := c.MakeVector(out)
	diagonal.AddScalar(c.MakeNumeric(scale))

	res.StateWeights.Vector.Scale(c.MakeNumeric(0))
	mapper.MapTranspose(diagonal, res.StateWeights.Vector)

	return res
}
