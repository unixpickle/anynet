package anyconv

import (
	"math"
	"testing"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestPostTrainer(t *testing.T) {
	net := anynet.Net{
		anynet.NewFC(anyvec64.CurrentCreator(), 3, 4),
		randomizedBatchNorm(2),
		&Residual{
			Projection: anynet.Net{
				anynet.NewFC(anyvec64.CurrentCreator(), 4, 5),
				randomizedBatchNorm(1),
			},
			Layer: anynet.Net{
				anynet.NewFC(anyvec64.CurrentCreator(), 4, 5),
				randomizedBatchNorm(5),
			},
		},
		&Residual{Layer: randomizedBatchNorm(5)},
	}

	var samples anyff.SliceSampleList
	for i := 0; i < 8; i++ {
		inVec := anyvec64.MakeVector(3)
		anyvec.Rand(inVec, anyvec.Normal, nil)
		samples = append(samples, &anyff.Sample{
			Input:  inVec,
			Output: anyvec64.MakeVector(5),
		})
	}

	fetcher := &anyff.Trainer{}
	fullBatch, _ := fetcher.Fetch(samples)

	expected := net.Apply(fullBatch.(*anyff.Batch).Inputs, samples.Len()).Output()

	pt := &PostTrainer{
		Samples:   samples,
		Fetcher:   fetcher,
		BatchSize: 3,
		Net:       net,
	}
	if err := pt.Run(); err != nil {
		t.Fatal(err)
	}

	if _, ok := net[1].(*BatchNorm); ok {
		t.Error("first BatchNorm stayed")
	}
	resid := net[2].(*Residual)
	for i, subLayer := range []anynet.Layer{resid.Layer, resid.Projection} {
		for j, layer := range subLayer.(anynet.Net) {
			if _, ok := layer.(*BatchNorm); ok {
				t.Errorf("residual part %d: layer %d is BatchNorm", i, j)
			}
		}
	}
	if _, ok := net[3].(*Residual).Layer.(*BatchNorm); ok {
		t.Error("second residual's BatchNorm stayed")
	}

	actual := net.Apply(fullBatch.(*anyff.Batch).Inputs, samples.Len()).Output()

	for i, x := range expected.Data().([]float64) {
		a := actual.Data().([]float64)[i]
		if math.Abs(x-a) > 1e-5 || math.IsNaN(a) || math.IsNaN(x) {
			t.Errorf("output %d should be %f but got %f", i, x, a)
		}
	}
}

func randomizedBatchNorm(inCount int) *BatchNorm {
	res := NewBatchNorm(anyvec64.CurrentCreator(), inCount)
	anyvec.Rand(res.Scalers.Vector, anyvec.Normal, nil)
	anyvec.Rand(res.Biases.Vector, anyvec.Normal, nil)
	return res
}
