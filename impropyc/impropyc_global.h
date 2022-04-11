#ifndef IMPROPYC_GLOBAL_H
#define IMPROPYC_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(IMPROPYC_LIBRARY)
#  define IMPROPYC_EXPORT Q_DECL_EXPORT
#else
#  define IMPROPYC_EXPORT Q_DECL_IMPORT
#endif

#endif // IMPROPYC_GLOBAL_H
