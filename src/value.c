#include "value.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

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
    return mgc_val_create(pow(b->value, p->value), 0.0, b, NULL, MGC_OP_POW);
}

struct mgc_val *mgc_neg(struct mgc_val *v)
{
    struct mgc_val *minus_one =
        mgc_val_create(-1.0, 0.0, NULL, NULL, MGC_OP_NONE);
    return mgc_mul(minus_one, v);
}

void mgc_backward(struct mgc_val *v)
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
        v->p0->grad += pow(v->p1->value, v->p0->value - 1) * v->grad;
        break;

    default:
        break;
    }
}
