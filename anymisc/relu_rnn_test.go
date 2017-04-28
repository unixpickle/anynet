package anymisc

import (
	"reflect"
	"testing"

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
