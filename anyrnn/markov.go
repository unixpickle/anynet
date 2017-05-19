package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var m Markov
	serializer.RegisterTypedDeserializer(m.SerializerType(), DeserializeMarkov)
}

// Markov is an RNN block that tracks a history of inputs.
type Markov struct {
	StartState *anydiff.Var

	// HistorySize is the number of inputs to store.
	// Since this does not include the input, the output
	// will contain HistorySize+1 packed inputs.
	HistorySize int

	// DepthWise controls how vectors from the history
	// are concatenated.
	// If false, then vectors a and b are joined to
	//
	//     <a1, a2, ..., b1, b2, ...>
	//
	// If true, then vectors are joined depth-wise
	//
	//     <a1, b1, a2, b2, ...>
	//
	// In either case, the first component of the most
	// recent timestep is packed before the first
	// component of the second most recent timestep.
	DepthWise bool
}

// NewMarkov creates a Markov with the history size and
// pre-known input vector size.
func NewMarkov(c anyvec.Creator, history int, inSize int) *Markov {
	if inSize == 0 || history == 0 {
		panic("input and history sizes must be non-zero")
	}
	return &Markov{
		StartState:  anydiff.NewVar(c.MakeVector(history * inSize)),
		HistorySize: history,
	}
}

// DeserializeMarkov deserializes a Markov.
func DeserializeMarkov(d []byte) (*Markov, error) {
	var res Markov
	var vec *anyvecsave.S
	err := serializer.DeserializeAny(d, &vec, &res.HistorySize, &res.DepthWise)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Markov", err)
	}
	res.StartState = anydiff.NewVar(vec.Vector)
	return &res, nil
}

// Start returns a start state.
func (m *Markov) Start(n int) State {
	return m.funcBlock().Start(n)
}

// PropagateStart propagates through the start state.
func (m *Markov) PropagateStart(sg StateGrad, g anydiff.Grad) {
	m.funcBlock().PropagateStart(sg, g)
}

// Step applies the block, returning a stacked tensor and
// updating the frame history in the state.
func (m *Markov) Step(state State, in anyvec.Vector) Res {
	return m.funcBlock().Step(state, in)
}

// Parameters returns the Markov's parameters.
func (m *Markov) Parameters() []*anydiff.Var {
	return []*anydiff.Var{m.StartState}
}

// SerializerType returns the unique ID used to serialize
// a Markov with the serializer package.
func (m *Markov) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.Markov"
}

// Serialize serializes the Markov.
func (m *Markov) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		&anyvecsave.S{Vector: m.StartState.Vector},
		m.HistorySize,
		m.DepthWise,
	)
}

func (m *Markov) funcBlock() *FuncBlock {
	return &FuncBlock{
		Func: func(in, state anydiff.Res, n int) (out, newState anydiff.Res) {
			oldRows := &anydiff.Matrix{
				Data: state,
				Rows: m.HistorySize,
				Cols: state.Output().Len() / m.HistorySize,
			}

			outRows := *oldRows
			outRows.Data = anydiff.Concat(in, oldRows.Data)
			outRows.Rows++

			stateRows := outRows
			stateRows.Rows--
			stateRows.Data = anydiff.Slice(outRows.Data, 0,
				stateRows.Cols*stateRows.Rows)

			if m.DepthWise {
				return anydiff.Transpose(&outRows).Data, stateRows.Data
			} else {
				return outRows.Data, stateRows.Data
			}
		},
		MakeStart: func(n int) anydiff.Res {
			var rep []anydiff.Res
			for i := 0; i < n; i++ {
				rep = append(rep, m.StartState)
			}
			return anydiff.Concat(rep...)
		},
	}
}
