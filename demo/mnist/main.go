package main

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/mnist"
)

var Creator anyvec.Creator

func main() {
	Creator = anyvec32.CurrentCreator()

	samples := trainingSet()
	network := anynet.Net{
		anynet.NewFC(Creator, 28*28, 300),
		anynet.Tanh,
		anynet.NewFC(Creator, 300, 10),
		anynet.LogSoftmax,
	}
}

func trainingSet() anyff.SampleList {
	ts := mnist.LoadTrainingDataSet()
	slice := make(anyff.SliceSampleList, len(ts.Samples))

	intensities := ts.IntensityVectors()
	labels := ts.LabelVectors()

	for i, intens := range intensities {
		label := labels[i]
		inVec := Creator.MakeVectorData(Creator.MakeNumericList(intens))
		outVec := Creator.MakeVectorData(Creator.MakeNumericList(label))
		slice[i] = &anyff.Sample{Input: inVec, Output: outVec}
	}

	return slice
}
