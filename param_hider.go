package anynet

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var p ParamHider
	serializer.RegisterTypedDeserializer(p.SerializerType(), DeserializeParamHider)
}

// A ParamHider wraps a Layer and does not implement
// Parameterizer, thus effectively freezing the parameters
// of the layer.
type ParamHider struct {
	Layer Layer
}

// DeserializeParamHider deserializes a ParamHider.
func DeserializeParamHider(d []byte) (*ParamHider, error) {
	var p ParamHider
	if err := serializer.DeserializeAny(d, &p.Layer); err != nil {
		return nil, essentials.AddCtx("deserialize ParamHider", err)
	}
	return &p, nil
}

// Apply applies the layer.
func (p *ParamHider) Apply(in anydiff.Res, n int) anydiff.Res {
	return p.Apply(in, n)
}

// SerializerType returns the unique ID used to serialize
// a ParamHider with the serializer package.
func (p *ParamHider) SerializerType() string {
	return "github.com/unixpickle/anynet.ParamHider"
}

// Serialize serializes the ParamHider.
func (p *ParamHider) Serialize() ([]byte, error) {
	return serializer.SerializeAny(p.Layer)
}
