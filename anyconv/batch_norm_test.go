package anyconv

import (
	"math"
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestBatchNormSerialize(t *testing.T) {
	layer := randomizedBatchNorm(4)
	data, err := serializer.SerializeAny(layer)
	if err != nil {
		t.Fatal(err)
	}
	var newLayer *BatchNorm
	if err := serializer.DeserializeAny(data, &newLayer); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(layer, newLayer) {
		t.Error("layers differ")
	}
}

func TestBatchNormOutput(t *testing.T) {
	layer := &BatchNorm{
		InputCount: 2,
		Scalers:    anydiff.NewVar(anyvec32.MakeVectorData([]float32{2, -3})),
		Biases:     anydiff.NewVar(anyvec32.MakeVectorData([]float32{-1.5, 2})),
	}
	vec := anyvec32.MakeVectorData([]float32{
		-0.636299517987754, 1.381820934572628, 1.117062796520384,
		-1.032042307499387, -0.603144099627179, 0.937477768422949,
	})
	actual := layer.Apply(anydiff.NewConst(vec), 1).Output().Data().([]float32)
	expected := []float32{
		-2.953427612694010, -0.723517206628873, 1.325934113323319,
		6.176822169129221, -2.872506500629310, 0.546695037499651,
	}
	for i, x := range expected {
		a := actual[i]
		if math.IsNaN(float64(a)) || math.Abs(float64(a-x)) > 1e-3 {
			t.Fatalf("expected %v but got %v", expected, actual)
		}
	}
}

func TestBatchNormProp(t *testing.T) {
	layer := NewBatchNorm(anyvec32.CurrentCreator(), 2)
	input := anyvec32.MakeVector(24)
	anyvec.Rand(input, anyvec.Normal, nil)
	inVar := anydiff.NewVar(input)

	checker := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return layer.Apply(inVar, 12)
		},
		V: []*anydiff.Var{inVar},
	}
	checker.FullCheck(t)
}
