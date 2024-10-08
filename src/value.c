#include "value.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static char *mgc_op_str[MGC_NOPS] = {"leaf", "+", "*", "^", "relu"};

static struct mgc_val *mgc_val_create(double value, double grad,
                                      struct mgc_val *p0, struct mgc_val *p1,
                                      enum mgc_op op)
{
    struct mgc_val *v = malloc(sizeof(struct mgc_val));
    if (v) {
        *v = (struct mgc_val){
            .value = value, .grad = grad, .p0 = p0, .p1 = p1, .op = op};
    }
    return v;
}

static void mgc_val_destroy(struct mgc_val *v) { free(v); }

struct mgc_val *mgc_value(double d)
{
    return mgc_val_create(d, 0.0, NULL, NULL, MGC_OP_NONE);
}

void mgc_value_free(struct mgc_val *v) { mgc_val_destroy(v); }

struct mgc_val *mgc_add(struct mgc_val *l, struct mgc_val *r)
{
    struct mgc_val *out =
        mgc_val_create(l->value + r->value, 0.0, l, r, MGC_OP_ADD);
    return out;
}

struct mgc_val *mgc_mul(struct mgc_val *l, struct mgc_val *r)
{
    struct mgc_val *out =
        mgc_val_create(l->value * r->value, 0.0, l, r, MGC_OP_MUL);
    return out;
}

struct mgc_val *mgc_pow(struct mgc_val *b, struct mgc_val *p)
{
    assert(p->op == MGC_OP_NONE && "power needs to be a scalar");
    return mgc_val_create(pow(b->value, p->value), 0.0, b, p, MGC_OP_POW);
}

struct mgc_val *mgc_neg(struct mgc_val *v)
{
    struct mgc_val *minus_one =
        mgc_val_create(-1.0, 0.0, NULL, NULL, MGC_OP_NONE);
    return mgc_mul(minus_one, v);
}

struct mgc_val *mgc_relu(struct mgc_val *v)
{
    return mgc_val_create(v->value > 0.0 ? v->value : 0.0, 0.0, v, NULL,
                          MGC_OP_RELU);
}

static void mgc_forward_step(struct mgc_val *v)
{
    switch (v->op) {
    case MGC_OP_NONE:
        break;

    case MGC_OP_ADD:
        v->value = v->p0->value + v->p1->value;
        break;

    case MGC_OP_MUL:
        v->value = v->p0->value * v->p1->value;
        break;

    case MGC_OP_POW:
        v->value = pow(v->p0->value, v->p1->value);
        break;

    case MGC_OP_RELU:
        v->value = v->p0->value > 0.0 ? v->p0->value : 0.0;

    default:
        break;
    }
}

void mgc_forward(struct mgc_ref_vec *sorted)
{
    ptrdiff_t size = mgc_ref_vec_size(sorted);
    for (ptrdiff_t i = size - 1; i >= 0; --i) {
        mgc_forward_step(mgc_ref_vec_at(sorted, i));
    }
}

void mgc_zero_gradient(struct mgc_ref_vec *sorted)
{
    ptrdiff_t size = mgc_ref_vec_size(sorted);
    for (ptrdiff_t i = 0; i < size; ++i) {
        mgc_ref_vec_at(sorted, i)->grad = 0.0;
    }
}

void mgc_sgd(struct mgc_ref_vec *params, ptrdiff_t size, double step)
{
    for (ptrdiff_t i = 0; i < size; ++i) {
        struct mgc_val *p = mgc_ref_vec_at(params, i);
        p->value -= p->grad * step;
    }
}

static void mgc_backward_step(struct mgc_val *v)
{
    switch (v->op) {
    case MGC_OP_NONE:
        break;

    case MGC_OP_ADD:
        v->p0->grad += v->grad;
        v->p1->grad += v->grad;
        break;

    case MGC_OP_MUL:
        v->p0->grad += v->p1->value * v->grad;
        v->p1->grad += v->p0->value * v->grad;
        break;

    case MGC_OP_POW:
        v->p0->grad +=
            v->p1->value * pow(v->p0->value, v->p1->value - 1) * v->grad;
        break;

    case MGC_OP_RELU:
        v->p0->grad += v->value > 0.0 ? v->grad : 0.0;

    default:
        break;
    }
}

bool mgc_ref_vec_contains(struct mgc_ref_vec *vec, struct mgc_val *m)
{
    /* This is O(n); replace with hash map at some point */
    for (ptrdiff_t i = 0; i < vec->size; ++i) {
        if (vec->refs[i] == m)
            return true;
    }
    return false;
}

static void toposort_recursive(struct mgc_ref_vec *sorted, struct mgc_val *v)
{
    if (!mgc_ref_vec_contains(sorted, v)) {
        mgc_ref_vec_append(sorted, v);
    }
    if (v->p0) {
        toposort_recursive(sorted, v->p0);
    }
    if (v->p1) {
        toposort_recursive(sorted, v->p1);
    }
}

ptrdiff_t mgc_toposort(struct mgc_ref_vec *sorted, struct mgc_val *v)
{
    toposort_recursive(sorted, v);
    return mgc_ref_vec_size(sorted);
}

void mgc_backward(struct mgc_ref_vec *sorted)
{
    for (ptrdiff_t i = 0; i < mgc_ref_vec_size(sorted); ++i) {
        mgc_backward_step(mgc_ref_vec_at(sorted, i));
    }
}

void mgc_print(struct mgc_val *v, ptrdiff_t depth)
{
    printf("%*s|-op=%s, val=%g, grad = %g\n", (int)depth, "", mgc_op_str[v->op],
           v->value, v->grad);
}

static void mgc_print_graph_recursive(struct mgc_val *v, ptrdiff_t depth)
{
    if (v) {
        mgc_print(v, depth);
        mgc_print_graph_recursive(v->p0, depth + 2);
        mgc_print_graph_recursive(v->p1, depth + 2);
    }
}

void mgc_print_graph(struct mgc_val *v) { mgc_print_graph_recursive(v, 0); }

struct mgc_ref_vec *mgc_ref_vec_init(struct mgc_ref_vec *vec)
{
    vec->capacity = 16;
    vec->size = 0;
    vec->refs = calloc(16, sizeof(struct mgc_val *));
    return vec;
}

struct mgc_ref_vec *mgc_ref_vec_fini(struct mgc_ref_vec *vec)
{
    vec->capacity = 0;
    vec->size = 0;
    free(vec->refs);
    vec->refs = 0;
    return vec;
}

ptrdiff_t mgc_ref_vec_size(struct mgc_ref_vec *vec) { return vec->size; }

struct mgc_ref_vec *mgc_ref_vec_append(struct mgc_ref_vec *vec,
                                       struct mgc_val *val)
{
    /* this will fail miserably if the allocation fails */
    if (vec->capacity <= vec->size) {
        vec->refs = (struct mgc_val **)realloc(
            vec->refs, vec->capacity * 2 * sizeof(struct mgc_val *));
    }
    vec->refs[vec->size] = val;
    vec->size++;
    return vec;
}

struct mgc_val *mgc_ref_vec_at(struct mgc_ref_vec *vec, ptrdiff_t index)
{
    return vec->refs[index];
}
