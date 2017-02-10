package anyctc

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestBestLabels(t *testing.T) {
	var inputs = [][][]float32{
		{
			{-9.21034037197618, -0.000100005000333347},
			{-0.105360515657826, -2.302585092994046},
			{-9.21034037197618, -0.000100005000333347},
			{-0.105360515657826, -2.302585092994046},
			{-9.21034037197618, -0.000100005000333347},
			{-9.21034037197618, -0.000100005000333347},
		},
		{
			{-1.38155105579643e+01, -1.38155105579643e+01, -2.00000199994916e-06},
			// The first label is not more likely, but
			// after both timesteps it has a 0.64% chance
			// of being seen in at least one of the two
			// timesteps.
			{-0.916290731874155, -13.815510557964274, -0.510827290434046},
			{-0.916290731874155, -13.815510557964274, -0.510827290434046},
			{-1.38155105579643e+01, -1.38155105579643e+01, -2.00000199994916e-06},
			{-1.609437912434100, -0.693147180559945, -1.203972804325936},
		},
		{
			{-1.38155105579643e+01, -1.38155105579643e+01, -2.00000199994916e-06},
			{-0.916290731874155, -13.815510557964274, -0.510827290434046},
			{-1.38155105579643e+01, -1.38155105579643e+01, -2.00000199994916e-06},
			{-1.609437912434100, -0.693147180559945, -1.203972804325936},
		},
		{
			{-0.916290731874155, -13.815510557964274, -0.510827290434046},
			{-1.38155105579643e+01, -1.38155105579643e+01, -2.00000199994916e-06},
			{-1.609437912434100, -0.693147180559945, -1.203972804325936},
		},
	}

	inLists := make([][]anyvec.Vector, len(inputs))
	for i, x := range inputs {
		inLists[i] = make([]anyvec.Vector, len(x))
		for j, y := range x {
			inLists[i][j] = anyvec32.MakeVectorData(y)
		}
	}
	in := anyseq.ConstSeqList(anyvec32.CurrentCreator(), inLists[1:])
	smallIn := anyseq.ConstSeqList(anyvec32.CurrentCreator(), inLists[:1])

	var expected = [][]int{
		{0, 0},
		{0, 1},
		{1},
		{1},
	}

	for _, thresh := range []float64{-1e-2, -1e-3, -1e-6, -1e-10} {
		actual := append(BestLabels(smallIn, thresh), BestLabels(in, thresh)...)
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("thresh %e: expected %v but got %v", thresh, expected, actual)
		}
	}
}
