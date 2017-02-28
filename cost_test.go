package anynet

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestDotCost(t *testing.T) {
	testCost(t, DotCost{}, []float32{
		1, 0.5, 2,
		3, -1, 2,
	}, []float32{
		-1, -2, -3,
		-2, -3, -1,
	}, []float32{8, 5}, 2)
}

func TestMSE(t *testing.T) {
	testCost(t, MSE{}, []float32{
		1, 0.5, 2,
		3, -1, 2,
	}, []float32{
		-1, -2, -3,
		-2, -3, -1,
	}, []float32{11 + 3.0/4, 12 + 2.0/3}, 2)
}

func TestSigmoidCE(t *testing.T) {
	t.Run("Unaveraged", func(t *testing.T) {
		testCost(t, SigmoidCE{}, []float32{
			1, 0.6,
			0.2, 0,
		}, []float32{
			1, 0,
			2, -1,
		}, []float32{
			0.3132616875 + 0.6931471806,
			0.02538560221 + 1.7015424088 + 0.3132616875,
		}, 2)
	})
	t.Run("Averaged", func(t *testing.T) {
		testCost(t, SigmoidCE{Average: true}, []float32{
			1, 0.6, 0,
			0.2, 0, 0,
		}, []float32{
			1, 0, -50,
			2, -1, -50,
		}, []float32{
			(1.0 / 3) * (0.3132616875 + 0.6931471806),
			(1.0 / 3) * (0.02538560221 + 1.7015424088 + 0.3132616875),
		}, 2)
	})
}

func TestHinge(t *testing.T) {
	testCost(t, Hinge{}, []float32{
		1, -1, -1, 1, 1, 1, -1, -1,
	}, []float32{
		0.5, 1, -2, 0.9, -2, 2, -1.5, -0.9,
	}, []float32{
		2.5, 0.1, 3, 0.1,
	}, 4)
}

func TestMultiHinge(t *testing.T) {
	testCost(t, MultiHinge{}, []float32{
		0, 1, 0,
		0, 0, 1,
	}, []float32{
		1, 2.5, 2,
		-2, -5, -3,
	}, []float32{
		0.5, 2,
	}, 2)
}

func TestMultiHingeProp(t *testing.T) {
	v1 := anydiff.NewVar(anyvec32.MakeVectorData([]float32{1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}))
	v2 := anydiff.NewVar(anyvec32.MakeVectorData([]float32{0, 0, 1, 0, 1, 0, 0, 0}))

	checker := &anydifftest.ResChecker{
		F: func() anydiff.Res {
			return MultiHinge{}.Cost(v2, v1, 2)
		},
		V: []*anydiff.Var{v1, v2},
	}
	checker.FullCheck(t)
}

func testCost(t *testing.T, c Cost, desired, output, expected []float32, n int) {
	desiredRes := anydiff.NewConst(anyvec32.MakeVectorData(desired))
	outputRes := anydiff.NewConst(anyvec32.MakeVectorData(output))

	actual := c.Cost(desiredRes, outputRes, n).Output().Data().([]float32)

	for i, x := range expected {
		a := actual[i]
		if math.IsNaN(float64(a)) || math.Abs(float64(x-a)) > 1e-3 {
			t.Errorf("component %d: expected %f but got %f", i, x, a)
		}
	}
}
