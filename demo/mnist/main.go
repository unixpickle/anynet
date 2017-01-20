package main

import (
	"log"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/rip"
)

var Creator anyvec.Creator

func main() {
	log.Println("Setting up...")

	Creator = anyvec32.CurrentCreator()

	network := anynet.Net{
		anynet.NewFC(Creator, 28*28, 300),
		anynet.Tanh,
		anynet.NewFC(Creator, 300, 10),
		anynet.LogSoftmax,
	}

	g := &anyff.Gradienter{
		Net:    network,
		Cost:   anynet.DotCost{},
		Params: network.Parameters(),
	}

	var iterNum int
	s := &anysgd.SGD{
		Gradienter:  g,
		Transformer: &anysgd.Adam{},
		Samples:     trainingSet(),
		Rater:       anysgd.ConstRater(0.001),
		StatusFunc: func(s anysgd.SampleList) {
			log.Printf("iter %d: cost=%f", iterNum, g.LastCost)
			iterNum++
		},
		BatchSize: 100,
	}

	log.Println("Press ctrl+c once to stop...")
	s.Run(rip.NewRIP())

	log.Println("Computing statistics...")
	printStats(network)
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

func printStats(net anynet.Net) {
	ts := mnist.LoadTestingDataSet()
	cf := func(in []float64) int {
		vec := Creator.MakeVectorData(Creator.MakeNumericList(in))
		inRes := anydiff.NewConst(vec)
		res := net.Apply(inRes, 1).Output()
		return anyvec.MaxIndex(res)
	}
	log.Println("Validation:", ts.NumCorrect(cf))
	log.Println("Histogram:", ts.CorrectnessHistogram(cf))
}
