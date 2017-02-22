package anysgd

import "github.com/unixpickle/anydiff"

// Momentum implements SGD with momentum.
//
// The transformed gradient v is computed as
//
//     v := momentum * v + grad
type Momentum struct {
	Momentum float64
	rolling  anydiff.Grad
}

// Transform transforms the gradient using momentum.
//
// This is not thread-safe.
func (m *Momentum) Transform(g anydiff.Grad) anydiff.Grad {
	if m.rolling == nil {
		m.rolling = copyGrad(g)
		return g
	}
	for v, x := range m.rolling {
		x.Scale(x.Creator().MakeNumeric(m.Momentum))
		x.Add(g[v])
		g[v].Set(x)
	}
	return g
}
