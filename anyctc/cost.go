package anyctc

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

// Cost computes the cost for a batch of output sequences.
// The cost for each sequence is the negative log
// likelihood of the corresponding label.
//
// For a sequence, suppose that all of the labels are
// bounded between 0 and N.
// Then there should be N+1 outputs at each timestep.
// The first N outputs correspond to the N labels.
// The last output is the special "blank" symbol.
// The outputs are all in the log domain.
//
// The anyvec.Creator must use an anyvec.NumericList type
// []float32 or []float64.
// No other numeric types are supported.
func Cost(seqs anyseq.Seq, labels [][]int) anydiff.Res {
	if len(seqs.Output()) == 0 {
		return anydiff.NewConst(seqs.Creator().MakeVector(0))
	}
	return anydiff.Scale(pool(seqs, func(in [][]anydiff.Res) anydiff.Res {
		var res []anydiff.Res
		for i, x := range in {
			res = append(res, logLikelihood(seqs.Creator(), x, labels[i]))
		}
		return anydiff.Concat(res...)
	}), seqs.Creator().MakeNumeric(-1))
}

type poolRes struct {
	In      anyseq.Seq
	Pools   []*anydiff.Var
	Lengths []int
	Res     anydiff.Res
}

func pool(seqs anyseq.Seq, f func(in [][]anydiff.Res) anydiff.Res) anydiff.Res {
	rawData := anyseq.SeparateSeqs(seqs.Output())
	pools := make([]*anydiff.Var, len(rawData))
	splitPools := make([][]anydiff.Res, len(rawData))
	lengths := make([]int, len(rawData))
	for i, raw := range rawData {
		pools[i] = anydiff.NewVar(seqs.Creator().Concat(raw...))
		splitPools[i] = splitRes(pools[i], len(raw))
		lengths[i] = len(raw)
	}
	return &poolRes{
		In:      seqs,
		Pools:   pools,
		Lengths: lengths,
		Res:     f(splitPools),
	}
}

func (p *poolRes) Output() anyvec.Vector {
	return p.Res.Output()
}

func (p *poolRes) Vars() anydiff.VarSet {
	return p.In.Vars()
}

func (p *poolRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	for _, pvar := range p.Pools {
		g[pvar] = pvar.Vector.Creator().MakeVector(pvar.Vector.Len())
	}
	p.Res.Propagate(u, g)
	downstream := make([][]anyvec.Vector, len(p.Pools))
	for i, pvar := range p.Pools {
		downstream[i] = splitVec(g[pvar], p.Lengths[i])
		delete(g, p.Pools[i])
	}
	joinedU := anyseq.ConstSeqList(u.Creator(), downstream).Output()
	p.In.Propagate(joinedU, g)
}

func splitVec(vec anyvec.Vector, parts int) []anyvec.Vector {
	res := make([]anyvec.Vector, parts)
	chunkSize := vec.Len() / parts
	for i := range res {
		res[i] = vec.Slice(i*chunkSize, (i+1)*chunkSize)
	}
	return res
}

func splitRes(res anydiff.Res, parts int) []anydiff.Res {
	if parts == 0 {
		return nil
	}
	reses := make([]anydiff.Res, parts)
	chunkSize := res.Output().Len() / parts
	for i := range reses {
		reses[i] = anydiff.Slice(res, i*chunkSize, (i+1)*chunkSize)
	}
	return reses
}
