package anynet

import (
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/serializer"
)

func init() {
	var a Activation
	serializer.RegisterTypedDeserializer(a.SerializerType(), DeserializeActivation)
}

// An Activation is a standard activation function.
type Activation int

// These are standard activation function.
const (
	Tanh Activation = iota
	LogSoftmax
)

// DeserializeActivation deserializes an Activation.
func DeserializeActivation(d []byte) (Activation, error) {
	if len(d) != 1 {
		return 0, fmt.Errorf("data length (%d) should be 1", len(d))
	}
	a := Activation(d[0])
	if a > LogSoftmax {
		return 0, fmt.Errorf("unknown activation ID: %d", a)
	}
	return a, nil
}

// Apply applies the activation function.
func (a Activation) Apply(in anydiff.Res, n int) anydiff.Res {
	switch a {
	case Tanh:
		return anydiff.Tanh(in)
	case LogSoftmax:
		inLen := in.Output().Len()
		if inLen%n != 0 {
			panic("batch size must divide input length")
		}
		return anydiff.LogSoftmax(in, inLen/n)
	default:
		panic(fmt.Sprintf("unknown activation: %d", a))
	}
}

// SerializerType returns the unique ID used to serialize
// an Activation.
func (a Activation) SerializerType() string {
	return "github.com/unixpickle/anynet.Activation"
}

// Serialize serializes the activation.
func (a Activation) Serialize() ([]byte, error) {
	return []byte{byte(a)}, nil
}
