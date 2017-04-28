package anymisc

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestIRNN(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	rnn := NewIRNN(c, 5, 3, 2)
	expectedDiag := c.MakeVectorData([]float64{
		2, 0, 0,
		0, 2, 0,
		0, 0, 2,
	})
	if !reflect.DeepEqual(rnn.StateWeights.Vector, expectedDiag) {
		t.Errorf("expected %v but got %v", expectedDiag, rnn.StateWeights.Vector)
	}
}

func TestNPRNN(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	rnn := NewNPRNN(c, 5, 4)
	matrix := &anyvec.Matrix{
		Data: rnn.StateWeights.Vector,
		Rows: 4,
		Cols: 4,
	}
	for i := 0; i < 30; i++ {
		inVec := &anyvec.Matrix{
			Data: c.MakeVector(matrix.Cols),
			Rows: matrix.Cols,
			Cols: 1,
		}
		product := &anyvec.Matrix{
			Data: c.MakeVector(matrix.Rows),
			Rows: matrix.Rows,
			Cols: 1,
		}
		anyvec.Rand(inVec.Data, anyvec.Normal, nil)
		inMag := anyvec.Norm(inVec.Data)
		product.Product(false, false, c.MakeNumeric(1), matrix, inVec, c.MakeNumeric(0))
		outMag := anyvec.Norm(product.Data)
		if inMag.(float64) < outMag.(float64) {
			t.Error("an eigenvalue was too big")
		}
	}
}
