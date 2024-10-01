#include <stdlib.h>
#include "value.h"

static struct mgc_val* mgc_val_create(double value, double grad, struct mgc_val *p0, struct mgc_val *p1, enum mgc_op op)
{
    struct mgc_val *v = malloc(sizeof(struct mgc_val));
    if (v) {
      *v = (struct mgc_val) { .value = value, .grad = grad, .p0 = p0, .p1 = p1, .op = op };
    }
    return v;
}

static void mgc_val_destroy(struct mgc_val *v)
{
  free(v);
}

struct mgc_val* mgc_value(double d)
{
    return mgc_val_create(d, 0.0, NULL, NULL, MGC_OP_NONE);
}

void mgc_value_free(struct mgc_val *v)
{
  mgc_val_destroy(v);
}

