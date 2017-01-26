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

func TestMeanPoolSerialize(t *testing.T) {
	mp := &MeanPool{
		SpanX:       3,
		SpanY:       2,
		StrideX:     4,
		StrideY:     5,
		InputWidth:  15,
		InputHeight: 13,
		InputDepth:  4,
	}
	data, err := serializer.SerializeAny(mp)
	if err != nil {
		t.Fatal(err)
	}
	var newLayer *MeanPool
	if err := serializer.DeserializeAny(data, &newLayer); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(newLayer, mp) {
		t.Fatal("layers differ")
	}
}

func TestMeanPoolOutput(t *testing.T) {
	mp := &MeanPool{
		SpanX:       3,
		SpanY:       2,
		StrideX:     2,
		StrideY:     1,
		InputWidth:  15,
		InputHeight: 13,
		InputDepth:  4,
	}
	input := anyvec32.MakeVector(15 * 13 * 4 * 2)
	anyvec.Rand(input, anyvec.Normal, nil)

	expected := naiveMeanPool(mp, input.Data().([]float32)[:15*13*4])
	expected = append(expected, naiveMeanPool(mp, input.Data().([]float32)[15*13*4:])...)
	actual := mp.Apply(anydiff.NewConst(input), 2).Output().Data().([]float32)

	if len(actual) != len(expected) {
		t.Fatalf("expected length %d but got %d", len(expected), len(actual))
	}

	for i, x := range expected {
		a := actual[i]
		if math.Abs(float64(x-a)) > 1e-3 {
			t.Errorf("output %d: should be %f but got %f", i, x, a)
			break
		}
	}
}

func TestMeanPoolProp(t *testing.T) {
	layer := &MeanPool{
		SpanX:       3,
		SpanY:       2,
		StrideX:     2,
		StrideY:     2,
		InputWidth:  15,
		InputHeight: 13,
		InputDepth:  4,
	}
	img := anyvec32.MakeVector(15 * 13 * 4 * 2)
	anyvec.Rand(img, anyvec.Normal, nil)
	inVar := anydiff.NewVar(img)

	checker := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return layer.Apply(inVar, 2)
		},
		V: []*anydiff.Var{inVar},
	}
	checker.FullCheck(t)
}

func naiveMeanPool(m *MeanPool, img []float32) []float32 {
	var res []float32
	for y := 0; y+m.SpanY <= m.InputHeight; y += m.StrideY {
		for x := 0; x+m.SpanX <= m.InputWidth; x += m.StrideX {
			for z := 0; z < m.InputDepth; z++ {
				res = append(res, meanInRegion(m, x, y, z, img))
			}
		}
	}
	return res
}

func meanInRegion(m *MeanPool, x, y, z int, img []float32) float32 {
	var sum float32
	for subY := 0; subY < m.SpanY && subY+y < m.InputHeight; subY++ {
		for subX := 0; subX < m.SpanX && subX+x < m.InputWidth; subX++ {
			idx := ((subY+y)*m.InputWidth+subX+x)*m.InputDepth + z
			sum += img[idx]
		}
	}
	return sum / float32(m.SpanX*m.SpanY)
}
