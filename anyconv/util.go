package anyconv

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

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
