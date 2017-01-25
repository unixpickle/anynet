package anyconv

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestPaddingOutput(t *testing.T) {
	pl := &Padding{
		InputWidth:  3,
		InputHeight: 4,
		InputDepth:  2,

		PaddingTop:    1,
		PaddingBottom: 2,
		PaddingLeft:   3,
		PaddingRight:  1,
	}

	inTensor := anyvec32.MakeVectorData([]float32{
		3.868200, 1.104760, 0.360270, 0.046398, 0.800748, -0.579334,
		-0.540134, -0.095748, -0.240087, 0.298587, 0.018990, 0.481808,
		-0.656787, -0.061479, 1.997873, 0.108665, 1.788285, 0.222048,
		1.153895, 0.780207, -0.655182, 0.495345, -0.244460, -0.841344,
	})

	expected := []float32{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 3.868200, 1.104760, 0.360270, 0.046398, 0.800748, -0.579334, 0, 0,
		0, 0, 0, 0, 0, 0, -0.540134, -0.095748, -0.240087, 0.298587, 0.018990, 0.481808, 0, 0,
		0, 0, 0, 0, 0, 0, -0.656787, -0.061479, 1.997873, 0.108665, 1.788285, 0.222048, 0, 0,
		0, 0, 0, 0, 0, 0, 1.153895, 0.780207, -0.655182, 0.495345, -0.244460, -0.841344, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	actual := pl.Apply(anydiff.NewConst(inTensor), 1).Output().Data().([]float32)

	if len(actual) != len(expected) {
		t.Fatalf("len should be %d but got %d", len(expected), len(actual))
	}

	for i, x := range expected {
		a := actual[i]
		if math.IsNaN(float64(a)) || math.Abs(float64(a-x)) > 1e-3 {
			t.Errorf("value %d: should be %f but got %f", i, x, a)
		}
	}
}

func TestPaddingProp(t *testing.T) {
	layer := &Padding{
		InputWidth:  3,
		InputHeight: 4,
		InputDepth:  2,

		PaddingTop:    1,
		PaddingBottom: 2,
		PaddingLeft:   3,
		PaddingRight:  1,
	}
	img := anyvec32.MakeVector(3 * 4 * 2 * 2)
	anyvec.Rand(img, anyvec.Uniform, nil)
	inVar := anydiff.NewVar(img)

	checker := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return layer.Apply(inVar, 2)
		},
		V: []*anydiff.Var{inVar},
	}
	checker.FullCheck(t)
}
