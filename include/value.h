#ifndef __MGC_VALUE_H__
#define __MGC_VALUE_H__

#include <stddef.h>

enum mgc_op {
    MGC_OP_NONE,
    MGC_OP_ADD,
    MGC_OP_MUL,
    MGC_OP_POW,
    MGC_OP_RELU,
    MGC_NOPS
};

struct mgc_val;
struct mgc_val {
    double value;
    double grad;
    struct mgc_val *p0, *p1;
    enum mgc_op op;
};
typedef struct mgc_val val;

struct mgc_val *mgc_value(double d);
void mgc_value_free(struct mgc_val *v);
void mgc_print_graph(struct mgc_val *v);

struct mgc_val *mgc_add(struct mgc_val *l, struct mgc_val *r);
struct mgc_val *mgc_mul(struct mgc_val *l, struct mgc_val *r);
struct mgc_val *mgc_pow(struct mgc_val *b, struct mgc_val *p);
struct mgc_val *mgc_neg(struct mgc_val *v);
struct mgc_val *mgc_relu(struct mgc_val *v);

void mgc_forward(struct mgc_val **sorted, ptrdiff_t size);
void mgc_backward(struct mgc_val **sorted, ptrdiff_t size);
void mgc_zero_gradient(struct mgc_val **sorted, ptrdiff_t size);
ptrdiff_t mgc_toposort(struct mgc_val **sorted, struct mgc_val *v);
void mgc_sgd(struct mgc_val **sorted, ptrdiff_t size, double step);

#endif
