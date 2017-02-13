package anyctc

import (
	"math"
	"sort"

	"github.com/unixpickle/anydiff/anyseq"
)

// BestLabels produces the most likely labelings for the
// output sequences.
//
// The blankThresh argument specifies how greedy the
// search should be with respect to blank symbols.
// Typically, a value close to -1e-3 is sufficient.
// As an example, a blankThresh of -0.0001 means that any
// blank with probability greater than e^-0.0001 is
// treated as if it had a 100% probability.
//
// A blankThresh of zero is not recommended unless the
// input sequences are fairly short.
func BestLabels(seqs anyseq.Seq, blankThresh float64) [][]int {
	var res [][]int
	for _, seq := range anyseq.SeparateSeqs(batchesTo64(seqs.Output())) {
		floatSeq := make([][]float64, len(seq))
		for i, x := range seq {
			floatSeq[i] = x.Data().([]float64)
		}
		res = append(res, prefixSearch(floatSeq, blankThresh))
	}
	return res
}

func prefixSearch(seq [][]float64, blankThresh float64) []int {
	var subSeqs [][][]float64
	var subSeq [][]float64
	for _, x := range seq {
		if x[len(x)-1] > blankThresh {
			if len(subSeq) > 0 {
				subSeqs = append(subSeqs, subSeq)
				subSeq = nil
			}
		} else {
			subSeq = append(subSeq, x)
		}
	}
	if len(subSeq) > 0 {
		subSeqs = append(subSeqs, subSeq)
	}

	var res []int
	for _, sub := range subSeqs {
		startProb := &labelProb{NoBlank: math.Inf(-1)}
		subRes, _ := subPrefixSearch(sub, nil, startProb)
		res = append(res, subRes...)
	}
	return res
}

func subPrefixSearch(seq [][]float64, prefix []int, prob *labelProb) ([]int, *labelProb) {
	if len(seq) == 0 {
		return prefix, prob
	}

	extensions := allExtensions(seq[0], prefix, prob)
	sort.Sort(extensionSorter(extensions))

	bestProb := zeroLabelProb()
	bestSeq := []int{}
	for _, ext := range extensions {
		if ext.Prob.Total() > bestProb.Total() {
			extended := append(append([]int{}, prefix...), ext.Addition...)
			res, finalProb := subPrefixSearch(seq[1:], extended, ext.Prob)
			if finalProb.Total() > bestProb.Total() {
				bestProb = finalProb
				bestSeq = res
			}
		}
	}

	return bestSeq, bestProb
}

// labelProb represents the probability of a labeling,
// split up into the probability of the labeling without a
// trailing blank and with a trailing blank.
type labelProb struct {
	Blank   float64
	NoBlank float64
}

func zeroLabelProb() *labelProb {
	return &labelProb{Blank: math.Inf(-1), NoBlank: math.Inf(-1)}
}

func (l *labelProb) Total() float64 {
	return addLogs(l.Blank, l.NoBlank)
}

// possibleExtension represents a possible way to extend a
// labeling (during prefix search).
type possibleExtension struct {
	// Tokens added to the labeling by this extension.
	Addition []int

	// Probability of the extended labeling.
	Prob *labelProb
}

func allExtensions(next []float64, label []int, prob *labelProb) []*possibleExtension {
	var res []*possibleExtension
	for i, compProb := range next[:len(next)-1] {
		p := zeroLabelProb()
		if len(label) > 0 && i == label[len(label)-1] {
			p.NoBlank = compProb + prob.Blank
		} else {
			p.NoBlank = compProb + prob.Total()
		}
		res = append(res, &possibleExtension{Addition: []int{i}, Prob: p})
	}
	noChangeProb := &labelProb{
		Blank:   prob.Total() + next[len(next)-1],
		NoBlank: math.Inf(-1),
	}
	if len(label) > 0 {
		last := label[len(label)-1]
		noChangeProb.NoBlank = prob.NoBlank + next[last]
	}
	return append(res, &possibleExtension{Prob: noChangeProb})
}

// An extensionSorter sorts possible labeling extensions
// from most to least probable.
type extensionSorter []*possibleExtension

func (e extensionSorter) Len() int {
	return len(e)
}

func (e extensionSorter) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

func (e extensionSorter) Less(i, j int) bool {
	return e[i].Prob.Total() > e[j].Prob.Total()
}
