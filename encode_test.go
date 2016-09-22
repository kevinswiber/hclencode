package hclencode

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestBasicEncoding(t *testing.T) {
	person := struct {
		Name       string `hcl:"name"`
		Age        int    `hcl:"age"`
		IsCustomer bool   `hcl:"is_customer"`
	}{
		Name:       "Kevin",
		Age:        34,
		IsCustomer: true,
	}

	personHCL := `name = "Kevin"

age = 34

is_customer = true
`
	output, err := MarshalIndent(person)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(t, personHCL, string(output))
}

func TestNested(t *testing.T) {
	type pet struct {
		Name string `hcl:"name,key"`
		Age  int    `hcl:"age"`
	}
	person := struct {
		Name       string `hcl:"name"`
		Age        int    `hcl:"age"`
		IsCustomer bool   `hcl:"is_customer"`
		Pets       []*pet `hcl:"pet"`
	}{
		Name:       "Kevin",
		Age:        34,
		IsCustomer: true,
		Pets:       []*pet{&pet{Name: "Spot", Age: 4}},
	}

	personHCL := `name = "Kevin"

age = 34

is_customer = true

pet = {
  Spot = {
    age = 4
  }
}
`
	output, err := MarshalIndent(person)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(t, personHCL, string(output))
}

func TestNestedSkipEqual(t *testing.T) {
	type pet struct {
		Name string `hcl:"name,key,skipequal"`
		Age  int    `hcl:"age"`
	}
	person := struct {
		Name       string `hcl:"name"`
		Age        int    `hcl:"age"`
		IsCustomer bool   `hcl:"is_customer"`
		Pets       []*pet `hcl:"pet,skipequal"`
	}{
		Name:       "Kevin",
		Age:        34,
		IsCustomer: true,
		Pets:       []*pet{&pet{Name: "Spot", Age: 4}},
	}

	personHCL := `name = "Kevin"

age = 34

is_customer = true

pet {
  Spot {
    age = 4
  }
}
`
	output, err := MarshalIndent(person)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(t, personHCL, string(output))
}

func TestNestedSkipEqualMergeKey(t *testing.T) {
	type pet struct {
		Name string `hcl:"name,key,skipequal,string"`
		Age  int    `hcl:"age"`
	}
	person := struct {
		Name       string `hcl:"name"`
		Age        int    `hcl:"age"`
		IsCustomer bool   `hcl:"is_customer"`
		Pets       []*pet `hcl:"pet,skipequal,mergekey"`
	}{
		Name:       "Kevin",
		Age:        34,
		IsCustomer: true,
		Pets:       []*pet{&pet{Name: "Spot", Age: 4}},
	}

	personHCL := `name = "Kevin"

age = 34

is_customer = true

pet "Spot" {
  age = 4
}
`
	output, err := MarshalIndent(person)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(t, personHCL, string(output))
}

type property struct {
	Name  string `hcl:"name"`
	Value string `hcl:"value"`
}

func (p *property) MarshalHCL() ([]byte, error) {
	ret := "\"" + p.Value + "\""
	return []byte(ret), nil
}

func TestMarshalerType(t *testing.T) {
	type parent struct {
		Properties []*property `hcl:"properties"`
	}

	props := []*property{&property{Name: "prop", Value: "val"}}
	p := parent{Properties: props}

	pHCL := `properties = ["val"]
`
	output, err := MarshalIndent(p)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(t, pHCL, string(output))
}

type pair struct {
	Key   string `hcl:",key,pivot,string"`
	Value string `hcl:",value"`
}

func TestMapStructType(t *testing.T) {
	type parent struct {
		Dictionary []*pair `hcl:"properties,skipequal"`
	}

	props := []*pair{&pair{Key: "key", Value: "val"},
		&pair{Key: "key2", Value: "val2"}}
	p := parent{Dictionary: props}

	pHCL := `properties {
  "key" = "val"

  "key2" = "val2"
}
`
	output, err := MarshalIndent(p)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(t, pHCL, string(output))
}
