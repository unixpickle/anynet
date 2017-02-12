package anyconv

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestResizeOut(t *testing.T) {
	r := &Resize{
		Depth:     2,
		InWidth:   2,
		InHeight:  2,
		OutWidth:  3,
		OutHeight: 4,
	}
	img := anyvec32.MakeVectorData([]float32{
		0.5, 0.4, 0.3, 0.2,
		0.1, 0.9, 0.8, 0.7,
	})
	expected := []float32{
		0.5, 0.4, 0.4, 0.3, 0.3, 0.2,
		0.3666667, 0.5666667, 0.4166667, 0.4666667, 0.4666667, 0.3666667,
		0.2333333, 0.7333333, 0.4333333, 0.6333333, 0.6333333, 0.5333333,
		0.1, 0.9, 0.45, 0.8, 0.8, 0.7,
	}
	actual := r.Apply(anydiff.NewConst(img), 1).Output().Data().([]float32)
	if len(actual) != len(expected) {
		t.Fatalf("length should be %d but got %d", len(expected), len(actual))
	}
	for i, x := range expected {
		a := actual[i]
		if math.Abs(float64(x-a)) > 1e-4 {
			t.Errorf("value %d: should be %f but got %f", i, x, a)
		}
	}
}

func TestResizeProp(t *testing.T) {
	r := &Resize{
		Depth:     3,
		InWidth:   4,
		InHeight:  7,
		OutWidth:  6,
		OutHeight: 6,
	}
	img := anyvec32.MakeVector(4 * 7 * 3 * 2)
	anyvec.Rand(img, anyvec.Normal, nil)
	inVar := anydiff.NewVar(img)

	checker := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return r.Apply(inVar, 2)
		},
		V: []*anydiff.Var{inVar},
	}
	checker.FullCheck(t)
}
