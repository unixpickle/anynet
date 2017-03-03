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

func TestConvSerialize(t *testing.T) {
	conv := &Conv{
		FilterCount:  4,
		FilterWidth:  3,
		FilterHeight: 2,
		StrideX:      1,
		StrideY:      2,
		InputWidth:   10,
		InputHeight:  9,
		InputDepth:   2,
	}
	conv.InitRand(anyvec32.DefaultCreator{})
	conv.Conver = MakeDefaultConver(*conv)
	data, err := serializer.SerializeAny(conv)
	if err != nil {
		t.Fatal(err)
	}
	var newConv *Conv
	if err := serializer.DeserializeAny(data, &newConv); err != nil {
		t.Fatal(err)
	}
	if newConv.Conver == nil {
		t.Fatal("no conver set")
	}

	// Set for deep equal.
	newConv.Conver = conv.Conver
	if !reflect.DeepEqual(newConv, conv) {
		t.Fatal("layers differ")
	}
}

func TestConvOutput(t *testing.T) {
	layer := &Conv{
		FilterCount:  4,
		FilterWidth:  3,
		FilterHeight: 2,
		StrideX:      1,
		StrideY:      2,
		InputWidth:   10,
		InputHeight:  9,
		InputDepth:   2,
		Filters: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			-0.5918121, -1.3422776, -0.0537671, -0.7292797,
			1.1903561, -0.5599870, 1.6847371, 0.8495952,
			-1.5001017, -0.1822000, -1.9970264, 1.7850264,
			-0.5320939, 0.4724099, -0.2785905, -0.7895553,
			-0.5436258, -1.0271329, -0.7510397, 1.2868528,
			0.8232482, 0.4082520, 0.0250436, -0.3359673,
			-0.3113687, -0.6814418, 0.2163325, -1.8351645,
			0.3426097, -1.0884304, -0.0059350, 0.9975638,
			0.7169898, -0.6132800, 2.0587411, -1.1866666,
			-1.4547028, -1.0895123, 0.5075943, -0.5050339,
			0.2383824, 0.0343446, -0.7432283, 0.0873460,
			0.1203291, 0.2631992, 1.7215463, 0.3856416,
		})),
		Biases: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			-0.061039, -0.465844, 2.731102, -0.231307,
		})),
	}
	layer.Conver = MakeDefaultConver(*layer)
	img := anyvec32.MakeVector(10 * 9 * 2 * 2)
	anyvec.Rand(img, anyvec.Normal, nil)

	expected := naiveConvolution(layer, img.Data().([]float32)[:10*9*2])
	expected = append(expected, naiveConvolution(layer, img.Data().([]float32)[10*9*2:])...)
	actual := layer.Apply(anydiff.NewConst(img), 2).Output().Data().([]float32)

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

func TestConvProp(t *testing.T) {
	layer := &Conv{
		FilterCount:  4,
		FilterWidth:  3,
		FilterHeight: 2,
		StrideX:      1,
		StrideY:      2,
		InputWidth:   10,
		InputHeight:  9,
		InputDepth:   2,
		Filters: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			-0.5918121, -1.3422776, -0.0537671, -0.7292797,
			1.1903561, -0.5599870, 1.6847371, 0.8495952,
			-1.5001017, -0.1822000, -1.9970264, 1.7850264,
			-0.5320939, 0.4724099, -0.2785905, -0.7895553,
			-0.5436258, -1.0271329, -0.7510397, 1.2868528,
			0.8232482, 0.4082520, 0.0250436, -0.3359673,
			-0.3113687, -0.6814418, 0.2163325, -1.8351645,
			0.3426097, -1.0884304, -0.0059350, 0.9975638,
			0.7169898, -0.6132800, 2.0587411, -1.1866666,
			-1.4547028, -1.0895123, 0.5075943, -0.5050339,
			0.2383824, 0.0343446, -0.7432283, 0.0873460,
			0.1203291, 0.2631992, 1.7215463, 0.3856416,
		})),
		Biases: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			-0.061039, -0.465844, 2.731102, -0.231307,
		})),
	}
	layer.Conver = MakeDefaultConver(*layer)
	img := anyvec32.MakeVector(10 * 9 * 2 * 2)
	anyvec.Rand(img, anyvec.Normal, nil)
	inVar := anydiff.NewVar(img)

	checker := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return layer.Apply(inVar, 2)
		},
		V:     []*anydiff.Var{inVar, layer.Filters, layer.Biases},
		Delta: 1e-3,
		Prec:  5e-3,
	}
	checker.FullCheck(t)
}

func naiveConvolution(c *Conv, img []float32) []float32 {
	var filters [][]float32
	for i := 0; i < c.FilterCount; i++ {
		fSize := c.FilterWidth * c.FilterHeight * c.InputDepth
		fData := c.Filters.Vector.Data().([]float32)[fSize*i : fSize*(i+1)]
		filters = append(filters, fData)
	}

	var res []float32
	for y := 0; y+c.FilterHeight <= c.InputHeight; y += c.StrideY {
		for x := 0; x+c.FilterWidth <= c.InputWidth; x += c.StrideX {
			for _, f := range filters {
				res = append(res, naiveFilter(c, x, y, img, f))
			}
		}
	}

	biases := c.Biases.Vector.Data().([]float32)
	for i := range res {
		res[i] += biases[i%len(biases)]
	}
	return res
}

func naiveFilter(c *Conv, x, y int, img, filter []float32) float32 {
	var sum float32
	for subY := 0; subY < c.FilterHeight; subY++ {
		for subX := 0; subX < c.FilterWidth; subX++ {
			for subZ := 0; subZ < c.InputDepth; subZ++ {
				idx := ((subY+y)*c.InputWidth+subX+x)*c.InputDepth + subZ
				filterIdx := (subY*c.FilterWidth+subX)*c.InputDepth + subZ
				sum += filter[filterIdx] * img[idx]
			}
		}
	}
	return sum
}
