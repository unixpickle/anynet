package anynet

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestDotCost(t *testing.T) {
	testCost(t, DotCost{}, []float32{
		1, 0.5, 2,
		3, -1, 2,
	}, []float32{
		-1, -2, -3,
		-2, -3, -1,
	}, []float32{8, 5})
}

func TestMSE(t *testing.T) {
	testCost(t, MSE{}, []float32{
		1, 0.5, 2,
		3, -1, 2,
	}, []float32{
		-1, -2, -3,
		-2, -3, -1,
	}, []float32{11 + 3.0/4, 12 + 2.0/3})
}

func testCost(t *testing.T, c Cost, desired, output, expected []float32) {
	desiredRes := anydiff.NewConst(anyvec32.MakeVectorData(desired))
	outputRes := anydiff.NewConst(anyvec32.MakeVectorData(output))

	actual := c.Cost(desiredRes, outputRes, 2).Output().Data().([]float32)

	for i, x := range expected {
		a := actual[i]
		if math.IsNaN(float64(a)) || math.Abs(float64(x-a)) > 1e-3 {
			t.Errorf("component %d: expected %f but got %f", i, x, a)
		}
	}
}
