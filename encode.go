package hclencode

import (
	"bytes"
	"encoding"
	"encoding/base64"
	"github.com/hashicorp/hcl/hcl/printer"
	"math"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// Marshal returns a byte buffer of HCL from an object.
func Marshal(v interface{}) ([]byte, error) {
	e := encodeState{}
	if err := e.marshal(v); err != nil {
		return nil, err
	}
	b := e.Bytes()
	end := len(b) - 1
	b = b[1:end]
	return b, nil
}

// MarshalIndent returns a byte buffer of formatted HCL from an object.
func MarshalIndent(v interface{}) ([]byte, error) {
	e := encodeState{}
	if err := e.marshal(v); err != nil {
		return nil, err
	}
	b := e.Bytes()
	end := len(b) - 1
	b = b[1:end]
	output, err := printer.Format(b)
	if err != nil {
		return nil, err
	}
	return output, nil
}

// Marshaler is the interface implemented by types that
// can marshal themselves into valid HCL.
type Marshaler interface {
	MarshalHCL() ([]byte, error)
}

// An UnsupportedTypeError is returned by Marshal when attempting
// to encode an unsupported type.
type UnsupportedTypeError struct {
	Type reflect.Type
}

func (e *UnsupportedTypeError) Error() string {
	return "hcl: unsupported type: " + e.Type.String()
}

// An UnsupportedValueError is returned by Marshal when attempting
// to encode an unsupported value.
type UnsupportedValueError struct {
	Value reflect.Value
	Str   string
}

func (e *UnsupportedValueError) Error() string {
	return "hcl: unsupported value: " + e.Str
}

// MarshalerError contains the type of the value being marshalled.
type MarshalerError struct {
	Type reflect.Type
	Err  error
}

func (e *MarshalerError) Error() string {
	return "hcl: error calling MarshalHCL for type " + e.Type.String() + ": " + e.Err.Error()
}

var hex = "0123456789abcdef"

//var numberType = reflect.TypeOf(Number(""))

type encodeState struct {
	bytes.Buffer
	scratch  [64]byte
	keyCache []string
}

func (e *encodeState) marshal(v interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				panic(r)
			}

			if s, ok := r.(string); ok {
				panic(s)
			}
			err = r.(error)
		}
	}()

	e.reflectValue(reflect.ValueOf(v))
	return nil

}

func (e *encodeState) error(err error) {
	panic(err)
}

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

func (e *encodeState) reflectValue(v reflect.Value) {
	valueEncoder(v)(e, v, false)
}

type encoderFunc func(e *encodeState, v reflect.Value, quoted bool)

var encoderCache struct {
	sync.RWMutex
	m map[reflect.Type]encoderFunc
}

func valueEncoder(v reflect.Value) encoderFunc {
	if !v.IsValid() {
		return invalidValueEncoder
	}
	return typeEncoder(v.Type())
}

func typeEncoder(t reflect.Type) encoderFunc {
	encoderCache.RLock()
	f := encoderCache.m[t]
	encoderCache.RUnlock()

	if f != nil {
		return f
	}

	// To deal with recursive types, populate the map with an
	// indirect func before we build it. This type waits on the
	// real func (f) to be ready and then calls it. This indirect
	// func is only used for recursive types.
	encoderCache.Lock()
	if encoderCache.m == nil {
		encoderCache.m = make(map[reflect.Type]encoderFunc)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	encoderCache.m[t] = func(e *encodeState, v reflect.Value, quoted bool) {
		wg.Wait()
		f(e, v, quoted)
	}
	encoderCache.Unlock()

	// Compute fields without lock.
	// Might duplicate effort but won't hold other computations back.
	f = newTypeEncoder(t, true)
	wg.Done()
	encoderCache.Lock()
	encoderCache.m[t] = f
	encoderCache.Unlock()
	return f
}

var (
	marshalerType     = reflect.TypeOf(new(Marshaler)).Elem()
	textMarshalerType = reflect.TypeOf(new(encoding.TextMarshaler)).Elem()
)

// newTypeEncoder constructs an encoderFunc for a type.
// The returned encoder only checks CanAddr when allowAddr is true.
func newTypeEncoder(t reflect.Type, allowAddr bool) encoderFunc {
	if t.Implements(marshalerType) {
		return marshalerEncoder
	}

	if t.Kind() != reflect.Ptr && allowAddr {
		if reflect.PtrTo(t).Implements(marshalerType) {
			return newCondAddrEncoder(addrMarshalerEncoder, newTypeEncoder(t, false))
		}
	}

	if t.Implements(textMarshalerType) {
		return textMarshalerEncoder
	}

	if t.Kind() != reflect.Ptr && allowAddr {
		if reflect.PtrTo(t).Implements(textMarshalerType) {
			return newCondAddrEncoder(addrTextMarshalerEncoder, newTypeEncoder(t, false))
		}
	}

	switch t.Kind() {
	case reflect.Bool:
		return boolEncoder
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return intEncoder
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return uintEncoder
	case reflect.Float32:
		return float32Encoder
	case reflect.Float64:
		return float64Encoder
	case reflect.String:
		return stringEncoder
	case reflect.Interface:
		return interfaceEncoder
	case reflect.Struct:
		return newStructEncoder(t)
	case reflect.Map:
		return newMapEncoder(t)
	case reflect.Slice:
		return newSliceEncoder(t)
	case reflect.Array:
		return newArrayEncoder(t)
	case reflect.Ptr:
		return newPtrEncoder(t)
	default:
		return unsupportedTypeEncoder
	}
}

func invalidValueEncoder(e *encodeState, _ reflect.Value, _ bool) {
	_, _ = e.WriteString("{}")
}

func marshalerEncoder(e *encodeState, v reflect.Value, _ bool) {
	if v.Kind() == reflect.Ptr && v.IsNil() {
		_, _ = e.WriteString("{}")
		return
	}

	m := v.Interface().(Marshaler)
	b, err := m.MarshalHCL()
	if err != nil {
		e.error(&MarshalerError{v.Type(), err})
		return
	}
	_, _ = e.Write(b)
}

func addrMarshalerEncoder(e *encodeState, v reflect.Value, _ bool) {
	va := v.Addr()
	if va.IsNil() {
		_, _ = e.WriteString("{}")
		return
	}
	m := va.Interface().(Marshaler)
	b, err := m.MarshalHCL()
	if err != nil {
		e.error(&MarshalerError{v.Type(), err})
		return
	}
	_, _ = e.Write(b)

}

func textMarshalerEncoder(e *encodeState, v reflect.Value, _ bool) {
	va := v.Addr()
	if va.IsNil() {
		_, _ = e.WriteString("\"\"")
		return
	}
	m := va.Interface().(encoding.TextMarshaler)
	b, err := m.MarshalText()
	if err != nil {
		e.error(&MarshalerError{v.Type(), err})
		return
	}
	e.stringBytes(b)
}

func addrTextMarshalerEncoder(e *encodeState, v reflect.Value, _ bool) {
	if v.Kind() == reflect.Ptr && v.IsNil() {
		_, _ = e.WriteString("\"\"")
		return
	}
	m := v.Interface().(encoding.TextMarshaler)
	b, err := m.MarshalText()
	if err != nil {
		e.error(&MarshalerError{v.Type(), err})
		return
	}
	e.stringBytes(b)
}

func boolEncoder(e *encodeState, v reflect.Value, quoted bool) {
	if quoted {
		_ = e.WriteByte('"')
	}
	if v.Bool() {
		_, _ = e.WriteString("true")
	} else {
		_, _ = e.WriteString("false")
	}
	if quoted {
		_ = e.WriteByte('"')
	}
}

func intEncoder(e *encodeState, v reflect.Value, quoted bool) {
	b := strconv.AppendInt(e.scratch[:0], v.Int(), 10)
	if quoted {
		_ = e.WriteByte('"')
	}
	_, _ = e.Write(b)
	if quoted {
		_ = e.WriteByte('"')
	}
}

func uintEncoder(e *encodeState, v reflect.Value, quoted bool) {
	b := strconv.AppendUint(e.scratch[:0], v.Uint(), 10)
	if quoted {
		_ = e.WriteByte('"')
	}
	_, _ = e.Write(b)
	if quoted {
		_ = e.WriteByte('"')
	}
}

type floatEncoder int // number of bits

func (bits floatEncoder) encode(e *encodeState, v reflect.Value, quoted bool) {
	f := v.Float()
	if math.IsInf(f, 0) || math.IsNaN(f) {
		e.error(&UnsupportedValueError{v, strconv.FormatFloat(f, 'g', -1, int(bits))})
	}
	b := strconv.AppendFloat(e.scratch[:0], f, 'g', -1, int(bits))
	if quoted {
		_ = e.WriteByte('"')
	}
	_, _ = e.Write(b)
	if quoted {
		_ = e.WriteByte('"')
	}
}

var (
	float32Encoder = (floatEncoder(32)).encode
	float64Encoder = (floatEncoder(64)).encode
)

func stringEncoder(e *encodeState, v reflect.Value, quoted bool) {
	/*
		if v.Type() == numberType {
			numStr := v.String()
			// In Go1.5 the empty string encodes to "0", while this is not a valid number literal
			// we keep compatibility so check validity after this.
			if numStr == "" {
				numStr = "0" // Number's zero-val
			}
			if !isValidNumber(numStr) {
				e.error(fmt.Errorf("json: invalid number literal %q", numStr))
			}
			_, _ = e.WriteString(numStr)
			return
		}
	*/
	if quoted {
		sb, err := Marshal(v.String())
		if err != nil {
			e.error(err)
		}
		e.string(string(sb))
	} else {
		e.string(v.String())
	}
}

func interfaceEncoder(e *encodeState, v reflect.Value, quoted bool) {
	if v.IsNil() {
		_, _ = e.WriteString("{}")
		return
	}
	e.reflectValue(v.Elem())
}

func unsupportedTypeEncoder(e *encodeState, v reflect.Value, _ bool) {
	e.error(&UnsupportedTypeError{v.Type()})
}

type structEncoder struct {
	fields    []field
	fieldEncs []encoderFunc
}

func (se *structEncoder) encode(e *encodeState, v reflect.Value, quoted bool) {
	_, _ = e.WriteString("{")
	for i, f := range se.fields {
		fv := fieldByIndex(v, f.index)
		if !fv.IsValid() || (f.omitEmpty && isEmptyValue(fv)) {
			continue
		}
		if f.key {
			if len(e.keyCache) > 0 {
				e.keyCache = e.keyCache[:len(e.keyCache)-1]
			}
			continue
		}
		if !f.mergeKey {
			_, _ = e.WriteString(f.name)
		}
		if f.skipEqual {
			_ = e.WriteByte(' ')
		} else {
			_ = e.WriteByte('=')
		}
		if f.mergeKey {
			e.keyCache = append(e.keyCache, f.name)
		}
		se.fieldEncs[i](e, fv, f.quoted)
		if f.mergeKey {
			e.keyCache = e.keyCache[:len(e.keyCache)-1]
		}
		if i < len(se.fields)-1 {
			_ = e.WriteByte(',')
		}
	}
	_ = e.WriteByte('}')
}

func newStructEncoder(t reflect.Type) encoderFunc {
	fields := cachedTypeFields(t)
	se := &structEncoder{
		fields:    fields,
		fieldEncs: make([]encoderFunc, len(fields)),
	}
	for i, f := range fields {
		se.fieldEncs[i] = typeEncoder(typeByIndex(t, f.index))
	}
	return se.encode
}

type mapEncoder struct {
	elemEnc encoderFunc
}

func (menc *mapEncoder) encode(e *encodeState, v reflect.Value, _ bool) {
	if v.IsNil() {
		_, _ = e.WriteString("{}")
		return
	}
	_, _ = e.WriteString("{\n")
	var sv stringValues = v.MapKeys()
	sort.Sort(sv)
	for _, k := range sv {
		e.string(k.String())
		_ = e.WriteByte('=')
		menc.elemEnc(e, v.MapIndex(k), false)
		_ = e.WriteByte('\n')
	}
	_, _ = e.WriteString("}\n")
}

func newMapEncoder(t reflect.Type) encoderFunc {
	if t.Key().Kind() != reflect.String {
		return unsupportedTypeEncoder
	}
	me := &mapEncoder{typeEncoder(t.Elem())}
	return me.encode
}

func encodeByteSlice(e *encodeState, v reflect.Value, _ bool) {
	if v.IsNil() {
		_, _ = e.WriteString("[]")
		return
	}
	s := v.Bytes()
	_ = e.WriteByte('"')
	if len(s) < 1024 {
		// for small buffers, using Encode directly is much faster.
		dst := make([]byte, base64.StdEncoding.EncodedLen(len(s)))
		base64.StdEncoding.Encode(dst, s)
		_, _ = e.Write(dst)
	} else {
		// for large buffers, avoid unnecessary extra temporary
		// buffer space.
		enc := base64.NewEncoder(base64.StdEncoding, e)
		_, _ = enc.Write(s)
		_ = enc.Close()
	}
	_ = e.WriteByte('"')
}

// sliceEncoder just wraps an arrayEncoder, checking to make sure the value isn't nil.
type sliceEncoder struct {
	arrayEnc encoderFunc
}

func (se *sliceEncoder) encode(e *encodeState, v reflect.Value, _ bool) {
	if v.IsNil() {
		_, _ = e.WriteString("[]")
		return
	}
	se.arrayEnc(e, v, false)
}

func newSliceEncoder(t reflect.Type) encoderFunc {
	// Byte slices get special treatment; arrays don't.
	if t.Elem().Kind() == reflect.Uint8 {
		return encodeByteSlice
	}
	enc := &sliceEncoder{newArrayEncoder(t)}
	return enc.encode
}

type arrayEncoder struct {
	key       string
	value     string
	skipEqual bool
	quotedKey bool
	pivot     bool
	elemEnc   encoderFunc
}

func (ae *arrayEncoder) encode(e *encodeState, v reflect.Value, _ bool) {
	if ae.key == "" {
		_ = e.WriteByte('[')
	} else if len(e.keyCache) == 0 {
		_ = e.WriteByte('{')
	}

	n := v.Len()
	for i := 0; i < n; i++ {
		if i > 0 {
			_ = e.WriteByte(',')
		}
		if ae.key != "" {
			ft := v.Index(i)
			if ft.Kind() == reflect.Ptr {
				ft = ft.Elem()
			}
			k := ft.FieldByName(ae.key).String()
			if ae.quotedKey {
				k = "\"" + k + "\""
			}
			e.keyCache = append(e.keyCache, k)
			_, _ = e.WriteString(strings.Join(e.keyCache, " "))
			if ae.skipEqual {
				_ = e.WriteByte(' ')
			} else {
				_ = e.WriteByte('=')
			}
			if ae.pivot && ae.value != "" {
				elemEnc := typeEncoder(ft.FieldByName(ae.value).Type())
				elemEnc(e, ft.FieldByName(ae.value), false)
				e.keyCache = e.keyCache[:len(e.keyCache)-1]
			} else {
				ae.elemEnc(e, v.Index(i), false)
			}
		} else {
			ae.elemEnc(e, v.Index(i), false)
		}
	}
	if ae.key == "" {
		_ = e.WriteByte(']')
	} else if len(e.keyCache) == 0 {
		_ = e.WriteByte('}')
	}
}

func newArrayEncoder(t reflect.Type) encoderFunc {
	var key, value string
	var skipEqual, pivot, quotedKey bool
	var fields []field
	elemEnc := typeEncoder(t.Elem())

	if t.Elem().Kind() == reflect.Ptr {
		fields = cachedTypeFields(t.Elem().Elem())
	} else {
		fields = cachedTypeFields(t.Elem())
	}

	for _, f := range fields {
		if f.key {
			key = f.originalName
			skipEqual = f.skipEqual
			pivot = f.pivot
			quotedKey = f.quoted
			break
		}

	}

	if pivot {
		for _, f := range fields {
			if f.value {
				value = f.originalName
				break
			}
		}
	}

	enc := &arrayEncoder{
		key:       key,
		value:     value,
		skipEqual: skipEqual,
		quotedKey: quotedKey,
		pivot:     pivot,
		elemEnc:   elemEnc,
	}

	return enc.encode
}

type ptrEncoder struct {
	elemEnc encoderFunc
}

func (pe *ptrEncoder) encode(e *encodeState, v reflect.Value, quoted bool) {
	if v.IsNil() {
		_, _ = e.WriteString("{}")
		return
	}
	pe.elemEnc(e, v.Elem(), quoted)
}

func newPtrEncoder(t reflect.Type) encoderFunc {
	enc := &ptrEncoder{typeEncoder(t.Elem())}
	return enc.encode
}

type condAddrEncoder struct {
	canAddrEnc, elseEnc encoderFunc
}

func (ce *condAddrEncoder) encode(e *encodeState, v reflect.Value, quoted bool) {

	if v.CanAddr() {

		ce.canAddrEnc(e, v, quoted)
	} else {
		ce.elseEnc(e, v, quoted)
	}
}

// newCondAddrEncoder returns an encoder that checks whether its value
// CanAddr and delegates to canAddrEnc if so, else to elseEnc.
func newCondAddrEncoder(canAddrEnc, elseEnc encoderFunc) encoderFunc {
	enc := &condAddrEncoder{canAddrEnc: canAddrEnc, elseEnc: elseEnc}
	return enc.encode
}

func isValidTag(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		switch {
		case strings.ContainsRune("!#$%&()*+-./:<=>?@[]^_{|}~ ", c):
			// Backslash and quote chars are reserved, but
			// otherwise any punctuation chars are allowed
			// in a tag name.
		default:
			if !unicode.IsLetter(c) && !unicode.IsDigit(c) {
				return false
			}
		}
	}
	return true
}

func fieldByIndex(v reflect.Value, index []int) reflect.Value {
	for _, i := range index {
		if v.Kind() == reflect.Ptr {
			if v.IsNil() {
				return reflect.Value{}
			}
			v = v.Elem()
		}
		v = v.Field(i)
	}
	return v
}

func typeByIndex(t reflect.Type, index []int) reflect.Type {
	for _, i := range index {
		if t.Kind() == reflect.Ptr {
			t = t.Elem()
		}
		t = t.Field(i).Type
	}
	return t
}

// stringValues is a slice of reflect.Value holding *reflect.StringValue.
// It implements the methods to sort by string.
type stringValues []reflect.Value

func (sv stringValues) Len() int           { return len(sv) }
func (sv stringValues) Swap(i, j int)      { sv[i], sv[j] = sv[j], sv[i] }
func (sv stringValues) Less(i, j int) bool { return sv.get(i) < sv.get(j) }
func (sv stringValues) get(i int) string   { return sv[i].String() }

// NOTE: keep in sync with stringBytes below.
func (e *encodeState) string(s string) int {
	len0 := e.Len()
	_ = e.WriteByte('"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if 0x20 <= b && b != '\\' && b != '"' {
				i++
				continue
			}
			if start < i {
				_, _ = e.WriteString(s[start:i])
			}
			switch b {
			case '\\', '"':
				_ = e.WriteByte('\\')
				_ = e.WriteByte(b)
			case '\n':
				_ = e.WriteByte('\\')
				_ = e.WriteByte('n')
			case '\r':
				_ = e.WriteByte('\\')
				_ = e.WriteByte('r')
			case '\t':
				_ = e.WriteByte('\\')
				_ = e.WriteByte('t')
			default:
				// This encodes bytes < 0x20 except for \n and \r,
				_, _ = e.WriteString(`\u00`)
				_ = e.WriteByte(hex[b>>4])
				_ = e.WriteByte(hex[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRuneInString(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				_, _ = e.WriteString(s[start:i])
			}
			_, _ = e.WriteString(`\ufffd`)
			i += size
			start = i
			continue
		}
	}

	if start < len(s) {
		_, _ = e.WriteString(s[start:])
	}
	_ = e.WriteByte('"')
	return e.Len() - len0
}

// NOTE: keep in sync with string above.
func (e *encodeState) stringBytes(s []byte) int {
	len0 := e.Len()
	_ = e.WriteByte('"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if 0x20 <= b && b != '\\' && b != '"' {
				i++
				continue
			}
			if start < i {
				_, _ = e.Write(s[start:i])
			}
			switch b {
			case '\\', '"':
				_ = e.WriteByte('\\')
				_ = e.WriteByte(b)
			case '\n':
				_ = e.WriteByte('\\')
				_ = e.WriteByte('n')
			case '\r':
				_ = e.WriteByte('\\')
				_ = e.WriteByte('r')
			case '\t':
				_ = e.WriteByte('\\')
				_ = e.WriteByte('t')
			default:
				// This encodes bytes < 0x20 except for \n and \r,
				_, _ = e.WriteString(`\u00`)
				_ = e.WriteByte(hex[b>>4])
				_ = e.WriteByte(hex[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRune(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				_, _ = e.Write(s[start:i])
			}
			_, _ = e.WriteString(`\ufffd`)
			i += size
			start = i
			continue
		}
	}

	if start < len(s) {
		_, _ = e.Write(s[start:])
	}
	_ = e.WriteByte('"')
	return e.Len() - len0
}

// A field represents a single field found in a struct.
type field struct {
	originalName string
	name         string
	nameBytes    []byte                 // []byte(name)
	equalFold    func(s, t []byte) bool // bytes.EqualFold or equivalent

	tag       bool
	index     []int
	typ       reflect.Type
	omitEmpty bool
	skipEqual bool
	pivot     bool
	key       bool
	value     bool
	mergeKey  bool
	quoted    bool
}

func fillField(f field) field {
	f.nameBytes = []byte(f.name)
	f.equalFold = foldFunc(f.nameBytes)
	return f
}

// byName sorts field by name, breaking ties with depth,
// then breaking ties with "name came from json tag", then
// breaking ties with index sequence.
type byName []field

func (x byName) Len() int { return len(x) }

func (x byName) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x byName) Less(i, j int) bool {
	if x[i].name != x[j].name {
		return x[i].name < x[j].name
	}
	if len(x[i].index) != len(x[j].index) {
		return len(x[i].index) < len(x[j].index)
	}
	if x[i].tag != x[j].tag {
		return x[i].tag
	}
	return byIndex(x).Less(i, j)
}

// byIndex sorts field by index sequence.
type byIndex []field

func (x byIndex) Len() int { return len(x) }

func (x byIndex) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x byIndex) Less(i, j int) bool {
	for k, xik := range x[i].index {
		if k >= len(x[j].index) {
			return false
		}
		if xik != x[j].index[k] {
			return xik < x[j].index[k]
		}
	}
	return len(x[i].index) < len(x[j].index)
}

// typeFields returns a list of fields that HCL should recognize for the given type.
// The algorithm is breadth-first search over the set of structs to include - the top struct
// and then any reachable anonymous structs.
func typeFields(t reflect.Type) []field {
	// Anonymous fields to explore at the current level and the next.
	current := []field{}
	next := []field{{typ: t}}

	// Count of queued names for current level and the next.
	count := map[reflect.Type]int{}
	nextCount := map[reflect.Type]int{}

	// Types already visited at an earlier level.
	visited := map[reflect.Type]bool{}

	// Fields found.
	var fields []field

	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, map[reflect.Type]int{}

		for _, f := range current {
			if visited[f.typ] {
				continue
			}
			visited[f.typ] = true

			// Scan f.typ for fields to include.
			for i := 0; i < f.typ.NumField(); i++ {
				sf := f.typ.Field(i)
				if sf.PkgPath != "" && !sf.Anonymous { // unexported
					continue
				}
				tag := sf.Tag.Get("hcl")
				if tag == "-" {
					continue
				}
				name, opts := parseTag(tag)
				if !isValidTag(name) {
					name = ""
				}
				index := make([]int, len(f.index)+1)
				copy(index, f.index)
				index[len(f.index)] = i

				ft := sf.Type
				if ft.Name() == "" && ft.Kind() == reflect.Ptr {
					// Follow pointer.
					ft = ft.Elem()
				}

				// Only strings, floats, integers, and booleans can be quoted.
				quoted := false
				if opts.Contains("string") {
					switch ft.Kind() {
					case reflect.Bool,
						reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
						reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
						reflect.Float32, reflect.Float64,
						reflect.String:
						quoted = true
					}
				}

				// Record found field and index sequence.
				if name != "" || !sf.Anonymous || ft.Kind() != reflect.Struct {
					tagged := name != ""
					if name == "" {
						name = sf.Name
					}
					fields = append(fields, fillField(field{
						originalName: sf.Name,
						name:         name,
						tag:          tagged,
						index:        index,
						typ:          ft,
						omitEmpty:    opts.Contains("omitempty"),
						skipEqual:    opts.Contains("skipequal"),
						pivot:        opts.Contains("pivot"),
						key:          opts.Contains("key"),
						value:        opts.Contains("value"),
						mergeKey:     opts.Contains("mergekey"),
						quoted:       quoted,
					}))
					if count[f.typ] > 1 {
						// If there were multiple instances, add a second,
						// so that the annihilation code will see a duplicate.
						// It only cares about the distinction between 1 or 2,
						// so don't bother generating any more copies.
						fields = append(fields, fields[len(fields)-1])
					}
					continue
				}

				// Record new anonymous struct to explore in next round.
				nextCount[ft]++
				if nextCount[ft] == 1 {
					next = append(next, fillField(field{name: ft.Name(), index: index, typ: ft}))
				}
			}
		}
	}

	sort.Sort(byName(fields))

	// Delete all fields that are hidden by the Go rules for embedded fields,
	// except that fields with HCL tags are promoted.

	// The fields are sorted in primary order of name, secondary order
	// of field index length. Loop over names; for each name, delete
	// hidden fields by choosing the one dominant field that survives.
	out := fields[:0]
	for advance, i := 0, 0; i < len(fields); i += advance {
		// One iteration per name.
		// Find the sequence of fields with the name of this first field.
		fi := fields[i]
		name := fi.name
		for advance = 1; i+advance < len(fields); advance++ {
			fj := fields[i+advance]
			if fj.name != name {
				break
			}
		}
		if advance == 1 { // Only one field with this name
			out = append(out, fi)
			continue
		}
		dominant, ok := dominantField(fields[i : i+advance])
		if ok {
			out = append(out, dominant)
		}
	}

	fields = out
	sort.Sort(byIndex(fields))

	return fields
}

// dominantField looks through the fields, all of which are known to
// have the same name, to find the single field that dominates the
// others using Go's embedding rules, modified by the presence of
// HCL tags. If there are multiple top-level fields, the boolean
// will be false: This condition is an error in Go and we skip all
// the fields.
func dominantField(fields []field) (field, bool) {
	// The fields are sorted in increasing index-length order. The winner
	// must therefore be one with the shortest index length. Drop all
	// longer entries, which is easy: just truncate the slice.
	length := len(fields[0].index)
	tagged := -1 // Index of first tagged field.
	for i, f := range fields {
		if len(f.index) > length {
			fields = fields[:i]
			break
		}
		if f.tag {
			if tagged >= 0 {
				// Multiple tagged fields at the same level: conflict.
				// Return no field.
				return field{}, false
			}
			tagged = i
		}
	}
	if tagged >= 0 {
		return fields[tagged], true
	}
	// All remaining fields have the same length. If there's more than one,
	// we have a conflict (two fields named "X" at the same level) and we
	// return no field.
	if len(fields) > 1 {
		return field{}, false
	}
	return fields[0], true
}

var fieldCache struct {
	sync.RWMutex
	m map[reflect.Type][]field
}

// cachedTypeFields is like typeFields but uses a cache to avoid repeated work.
func cachedTypeFields(t reflect.Type) []field {
	fieldCache.RLock()
	f := fieldCache.m[t]
	fieldCache.RUnlock()
	if f != nil {
		return f
	}

	// Compute fields without lock.
	// Might duplicate effort but won't hold other computations back.
	f = typeFields(t)
	if f == nil {
		f = []field{}
	}

	fieldCache.Lock()
	if fieldCache.m == nil {
		fieldCache.m = map[reflect.Type][]field{}
	}
	fieldCache.m[t] = f
	fieldCache.Unlock()
	return f
}
