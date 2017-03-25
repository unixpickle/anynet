package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Feedback{}).SerializerType(), DeserializeFeedback)
}

// Feedback is a block which feeds each output back in as
// input at the next timestep.
type Feedback struct {
	// Mixer combines the block input and the previous
	// output (in that order) to produce inputs for Block.
	Mixer anynet.Mixer

	// Block takes inputs from Mixer and produces outputs.
	Block Block

	// InitOut is fed to the Mixer at the first timestep
	// as if it were the previous output.
	InitOut *anydiff.Var
}

// DeserializeFeedback deserializes a Feedback block.
func DeserializeFeedback(d []byte) (*Feedback, error) {
	var res Feedback
	var initOut *anyvecsave.S
	err := serializer.DeserializeAny(d, &res.Mixer, &res.Block, &initOut)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Feedback", err)
	}
	res.InitOut = anydiff.NewVar(initOut.Vector)
	return &res, nil
}

// Start generates a start *FeedbackState.
func (f *Feedback) Start(n int) State {
	return &FeedbackState{
		BlockState: f.Block.Start(n),
		LastOut:    NewVecState(f.InitOut.Vector, n),
	}
}

// PropagateStart performs back-propagation through the
// start state.
func (f *Feedback) PropagateStart(s StateGrad, g anydiff.Grad) {
	sg := s.(*FeedbackGrad)
	f.Block.PropagateStart(sg.BlockGrad, g)
	sg.LastOut.PropagateStart(f.InitOut, g)
}

// Step applies the block.
func (f *Feedback) Step(s State, in anyvec.Vector) Res {
	fs := s.(*FeedbackState)
	inPool := anydiff.NewVar(in)
	lastPool := anydiff.NewVar(fs.LastOut.Vector)
	mixed := f.Mixer.Mix(inPool, lastPool, s.Present().NumPresent())
	blockOut := f.Block.Step(fs.BlockState, mixed.Output())
	v := anydiff.NewVarSet(f.InitOut)
	for _, p := range anynet.AllParameters(f.Mixer) {
		v.Add(p)
	}
	v = anydiff.MergeVarSets(v, blockOut.Vars())
	return &feedbackRes{
		InPool:      inPool,
		LastOutPool: lastPool,
		Mixed:       mixed,
		BlockRes:    blockOut,
		V:           v,
		OutState: &FeedbackState{
			BlockState: blockOut.State(),
			LastOut: &VecState{
				Vector:     blockOut.Output(),
				PresentMap: s.Present(),
			},
		},
	}
}

// Parameters returns the parameters of the block.
func (f *Feedback) Parameters() []*anydiff.Var {
	return append(anynet.AllParameters(f.Mixer, f.Block), f.InitOut)
}

// SerializerType returns the unique ID used to serialize
// a Feedback block with the serializer package.
func (f *Feedback) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.Feedback"
}

// Serialize serializes the block.
func (f *Feedback) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		f.Mixer,
		f.Block,
		&anyvecsave.S{Vector: f.InitOut.Vector},
	)
}

// FeedbackState is a State for a Feedback block.
type FeedbackState struct {
	BlockState State
	LastOut    *VecState
}

// Present returns the present map.
func (f *FeedbackState) Present() PresentMap {
	return f.BlockState.Present()
}

// Reduce reduces the state.
func (f *FeedbackState) Reduce(p PresentMap) State {
	return &FeedbackState{
		BlockState: f.BlockState.Reduce(p),
		LastOut:    f.LastOut.Reduce(p).(*VecState),
	}
}

// FeedbackGrad is the StateGrad for a Feedback block.
type FeedbackGrad struct {
	BlockGrad StateGrad
	LastOut   *VecState
}

// Present returns the present map.
func (f *FeedbackGrad) Present() PresentMap {
	return f.BlockGrad.Present()
}

// Expand expands the state.
func (f *FeedbackGrad) Expand(p PresentMap) StateGrad {
	return &FeedbackGrad{
		BlockGrad: f.BlockGrad.Expand(p),
		LastOut:   f.LastOut.Expand(p).(*VecState),
	}
}

type feedbackRes struct {
	InPool      *anydiff.Var
	LastOutPool *anydiff.Var
	Mixed       anydiff.Res
	BlockRes    Res
	V           anydiff.VarSet
	OutState    *FeedbackState
}

func (f *feedbackRes) State() State {
	return f.OutState
}

func (f *feedbackRes) Output() anyvec.Vector {
	return f.BlockRes.Output()
}

func (f *feedbackRes) Vars() anydiff.VarSet {
	return f.V
}

func (f *feedbackRes) Propagate(u anyvec.Vector, s StateGrad,
	g anydiff.Grad) (anyvec.Vector, StateGrad) {
	for _, p := range []*anydiff.Var{f.InPool, f.LastOutPool} {
		g[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
		defer func(g anydiff.Grad, p *anydiff.Var) {
			delete(g, p)
		}(g, p)
	}
	var blockUpstream StateGrad
	if s != nil {
		fs := s.(*FeedbackGrad)
		blockUpstream = fs.BlockGrad
		u.Add(fs.LastOut.Vector)
	}
	inDownstream, blockDown := f.BlockRes.Propagate(u, blockUpstream, g)
	f.Mixed.Propagate(inDownstream, g)
	return g[f.InPool], &FeedbackGrad{
		BlockGrad: blockDown,
		LastOut: &VecState{
			Vector:     g[f.LastOutPool],
			PresentMap: f.OutState.Present(),
		},
	}
}
