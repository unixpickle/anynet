package anyconv

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/serializer"
)

func init() {
	var r Residual
	serializer.RegisterTypedDeserializer(r.SerializerType(), DeserializeResidual)
}

// Residual implements residual shortcut connections.
type Residual struct {
	// Layer is the residual mapping.
	Layer anynet.Layer

	// If non-nil, Projection is applied to the original
	// input before it is added back to the output of Layer.
	//
	// This can be used to deal with residual mappings that
	// change the tensor dimensions.
	Projection anynet.Layer
}

// DeserializeResidual deserializes a Residual.
func DeserializeResidual(d []byte) (*Residual, error) {
	var layer anynet.Layer
	var proj anynet.Net
	if err := serializer.DeserializeAny(d, &layer, &proj); err != nil {
		return nil, err
	}
	res := &Residual{Layer: layer}
	if len(proj) == 1 {
		res.Projection = proj[0]
	}
	return res, nil
}

// Apply applies the layer.
func (r *Residual) Apply(in anydiff.Res, batch int) anydiff.Res {
	return anydiff.Pool(in, func(in anydiff.Res) anydiff.Res {
		mainOut := r.Layer.Apply(in, batch)
		orig := in
		if r.Projection != nil {
			orig = r.Projection.Apply(in, batch)
		}
		return anydiff.Add(orig, mainOut)
	})
}

// Parameters returns the joined parameters of the Layer
// and (if applicable) the Projection.
func (r *Residual) Parameters() []*anydiff.Var {
	n := anynet.Net{r.Layer}
	if r.Projection != nil {
		n = append(n, r.Projection)
	}
	return n.Parameters()
}

// SerializerType returns the unique ID used to serialize
// a Residual with the serializer package.
func (r *Residual) SerializerType() string {
	return "github.com/unixpickle/anynet/anyconv.Residual"
}

// Serialize serializes the Residual.
func (r *Residual) Serialize() ([]byte, error) {
	var projLayer anynet.Net
	if r.Projection != nil {
		projLayer = anynet.Net{r.Projection}
	}
	return serializer.SerializeAny(
		r.Layer,
		projLayer,
	)
}
