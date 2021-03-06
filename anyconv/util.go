package anyconv

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
)

// Weights returns all of the filters and weight matrices
// in the layer without returning biases, scalers, etc.
// It can extract anynet.Net and Residual layers.
func Weights(l anynet.Layer) []*anydiff.Var {
	var res []*anydiff.Var
	switch l := l.(type) {
	case anynet.Net:
		for _, sub := range l {
			res = append(res, Weights(sub)...)
		}
	case *anynet.FC:
		res = append(res, l.Weights)
	case *Conv:
		res = append(res, l.Filters)
	case *Residual:
		res = append(res, Weights(l.Layer)...)
		if l.Projection != nil {
			res = append(res, Weights(l.Projection)...)
		}
	}
	return res
}

type meanRowsRes struct {
	In     anydiff.Res
	Scaler anyvec.Numeric
	Out    anyvec.Vector
}

// negMeanRows computes the negative of the mean of the
// rows in a row-major matrix.
func negMeanRows(in anydiff.Res, cols int) anydiff.Res {
	if in.Output().Len()%cols != 0 {
		panic("column count must divide input size")
	}
	rows := in.Output().Len() / cols
	scaler := in.Output().Creator().MakeNumeric(-1 / float64(rows))
	out := anyvec.SumRows(in.Output().Copy(), cols)
	out.Scale(scaler)
	return &meanRowsRes{
		In:     in,
		Scaler: scaler,
		Out:    out,
	}
}

func (m *meanRowsRes) Output() anyvec.Vector {
	return m.Out
}

func (m *meanRowsRes) Vars() anydiff.VarSet {
	return m.In.Vars()
}

func (m *meanRowsRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	u.Scale(m.Scaler)
	if v, ok := m.In.(*anydiff.Var); ok {
		downstream, ok := g[v]
		if !ok {
			return
		}
		anyvec.AddRepeated(downstream, u)
	} else {
		downstream := m.Out.Creator().MakeVector(m.In.Output().Len())
		anyvec.AddRepeated(downstream, u)
		m.In.Propagate(downstream, g)
	}

}

type meanSquareRes struct {
	In     anydiff.Res
	Scaler anyvec.Numeric
	Out    anyvec.Vector
}

// meanSquare is like meanRows, but is squares the rows
// before taking their mean.
func meanSquare(in anydiff.Res, cols int) anydiff.Res {
	if in.Output().Len()%cols != 0 {
		panic("column count must divide input size")
	}
	rows := in.Output().Len() / cols
	scaler := in.Output().Creator().MakeNumeric(1 / float64(rows))
	squareIn := in.Output().Copy()
	squareIn.Mul(in.Output())
	out := anyvec.SumRows(squareIn, cols)
	out.Scale(scaler)
	return &meanSquareRes{
		In:     in,
		Scaler: in.Output().Creator().MakeNumeric(2 / float64(rows)),
		Out:    out,
	}
}

func (m *meanSquareRes) Output() anyvec.Vector {
	return m.Out
}

func (m *meanSquareRes) Vars() anydiff.VarSet {
	return m.In.Vars()
}

func (m *meanSquareRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	u.Scale(m.Scaler)
	downstream := m.Out.Creator().MakeVector(m.In.Output().Len())
	anyvec.AddRepeated(downstream, u)
	downstream.Mul(m.In.Output())
	m.In.Propagate(downstream, g)
}

func batchMap(m anyvec.Mapper, in anyvec.Vector) anyvec.Vector {
	var mapped []anyvec.Vector
	n := in.Len() / m.InSize()
	for i := 0; i < n; i++ {
		sub := in.Slice(m.InSize()*i, m.InSize()*(i+1))
		newSub := in.Creator().MakeVector(m.OutSize())
		m.Map(sub, newSub)
		mapped = append(mapped, newSub)
	}
	return in.Creator().Concat(mapped...)
}

func batchMapTranspose(m anyvec.Mapper, in anyvec.Vector) anyvec.Vector {
	var mapped []anyvec.Vector
	n := in.Len() / m.OutSize()
	for i := 0; i < n; i++ {
		sub := in.Slice(m.OutSize()*i, m.OutSize()*(i+1))
		newSub := in.Creator().MakeVector(m.InSize())
		m.MapTranspose(sub, newSub)
		mapped = append(mapped, newSub)
	}
	return in.Creator().Concat(mapped...)
}
