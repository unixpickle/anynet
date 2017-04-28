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
	res.StateWeights.Vector.Set(identityMatrix(c, out, scale).Data)
	return res
}

// NewNPRNN creates an RNN with ReLU activations and a
// positive-definite transition matrix with bounded
// eigenvalues.
//
// This is based on https://arxiv.org/abs/1511.03771.
func NewNPRNN(c anyvec.Creator, in, out int) *anyrnn.Vanilla {
	res := anyrnn.NewVanilla(c, in, out, anynet.ReLU)

	factor := &anyvec.Matrix{
		Data: c.MakeVector(out * out),
		Rows: out,
		Cols: out,
	}
	anyvec.Rand(factor.Data, anyvec.Normal, nil)
	posDef := identityMatrix(c, out, 1)
	posDef.Product(true, false, c.MakeNumeric(1/float64(out)), factor, factor,
		c.MakeNumeric(1))

	res.StateWeights.Vector.Set(posDef.Data)
	res.StateWeights.Vector.Scale(inverseLargestEig(posDef))

	return res
}

func identityMatrix(c anyvec.Creator, size int, scale float64) *anyvec.Matrix {
	hiddenIndices := make([]int, size)
	for i := range hiddenIndices {
		hiddenIndices[i] = i * (size + 1)
	}
	mapper := c.MakeMapper(size*size, hiddenIndices)
	diagonal := c.MakeVector(size)
	diagonal.AddScalar(c.MakeNumeric(scale))

	res := c.MakeVector(size * size)
	mapper.MapTranspose(diagonal, res)
	return &anyvec.Matrix{Data: res, Rows: size, Cols: size}
}

func inverseLargestEig(mat *anyvec.Matrix) anyvec.Numeric {
	const numIters = 100

	c := mat.Data.Creator()
	ops := c.NumOps()
	inVec := c.MakeVector(mat.Cols)
	outVec := c.MakeVector(mat.Cols)

	anyvec.Rand(inVec, anyvec.Normal, nil)

	// Power iteration method: it's slow, but it works.
	for i := 0; i < numIters; i++ {
		mag := anyvec.Norm(inVec)
		inVec.Scale(ops.Div(c.MakeNumeric(1), mag))
		anyvec.Gemv(false, mat.Rows, mat.Cols, c.MakeNumeric(1), mat.Data,
			mat.Cols, inVec, 1, c.MakeNumeric(0), outVec, 1)
		inVec, outVec = outVec, inVec
	}

	return ops.Div(c.MakeNumeric(1), anyvec.Norm(inVec))
}
