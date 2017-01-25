package anyconv

import (
	"math"
	"testing"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestPostTrainer(t *testing.T) {
	net := anynet.Net{
		anynet.NewFC(anyvec32.CurrentCreator(), 3, 4),
		randomizedBatchNorm(2),
		anynet.NewFC(anyvec32.CurrentCreator(), 4, 5),
		randomizedBatchNorm(5),
	}

	var samples anyff.SliceSampleList
	for i := 0; i < 8; i++ {
		inVec := anyvec32.MakeVector(3)
		anyvec.Rand(inVec, anyvec.Normal, nil)
		samples = append(samples, &anyff.Sample{
			Input:  inVec,
			Output: anyvec32.MakeVector(5),
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
	if _, ok := net[3].(*BatchNorm); ok {
		t.Error("second BatchNorm stayed")
	}

	actual := net.Apply(fullBatch.(*anyff.Batch).Inputs, samples.Len()).Output()

	for i, x := range expected.Data().([]float32) {
		a := actual.Data().([]float32)[i]
		if math.Abs(float64(x-a)) > 1e-3 || math.IsNaN(float64(a)) ||
			math.IsNaN(float64(x)) {
			t.Errorf("output %d should be %f but got %f", i, x, a)
		}
	}
}

func randomizedBatchNorm(inCount int) *BatchNorm {
	res := NewBatchNorm(anyvec32.CurrentCreator(), inCount)
	anyvec.Rand(res.Scalers.Vector, anyvec.Normal, nil)
	anyvec.Rand(res.Biases.Vector, anyvec.Normal, nil)
	return res
}
