#ifndef __KL2EDK_AUTOGEN_global__
#define __KL2EDK_AUTOGEN_global__

#ifdef KL2EDK_INCLUDE_MESSAGES
  #pragma message ( "Including 'global.h'" )
#endif

////////////////////////////////////////////////////////////////
// THIS FILE IS AUTOMATICALLY GENERATED -- DO NOT MODIFY!!
////////////////////////////////////////////////////////////////
// Generated by kl2edk version 1.15.2
////////////////////////////////////////////////////////////////

#include <FabricEDK.h>
#if FABRIC_EDK_VERSION_MAJ != 1 || FABRIC_EDK_VERSION_MIN != 15
# error "This file needs to be rebuilt for the current EDK version!"
#endif

// Dependencies on other extensions
#define FABRIC_EDK_EXT_MkMNIST_DEPENDENT_EXTS \
  { \
    { 0, 0, 0, 0, 0 } \
  }

// forward declarations
namespace Fabric { namespace EDK { namespace KL {
  class Object;
  class MkMNIST;
}}}

#include "aliases.h"

namespace Fabric { namespace EDK { namespace KL {

// KL interface 'Object'
// Defined at (internal)

class Object
{
public:
  
  struct VTable
  {
  };
  
  struct Bits
  {
    ObjectCore *objectCorePtr;
    SwapPtr<VTable const> const *vTableSwapPtrPtr;
  } *m_bits;
  
protected:
  
  friend struct Traits< Object >;
  
  static void ConstructEmpty( Object *self )
  {
    self->m_bits = 0;
  }
  
  static void ConstructCopy( Object *self, Object const *other )
  {
    if ( (self->m_bits = other->m_bits) )
      AtomicUInt32Increment( &self->m_bits->objectCorePtr->refCount );
  }
  
  static void AssignCopy( Object *self, Object const *other )
  {
    if ( self->m_bits != other->m_bits )
    {
      Destruct( self );
      ConstructCopy( self, other );
    }
  }
  
  static void Destruct( Object *self )
  {
    if ( self->m_bits
      && AtomicUInt32DecrementAndGetValue( &self->m_bits->objectCorePtr->refCount ) == 0 )
    {
      self->m_bits->objectCorePtr->lTableSwapPtrPtr->get()->lifecycleDestroy(
        &self->m_bits->objectCorePtr
        );
    }
  }
  
public: 
  
  typedef Object &Result;
  typedef Object const &INParam;
  typedef Object &IOParam;
  
  Object()
  {
    ConstructEmpty( this );
  }
  
  Object( Object const &that )
  {
    ConstructCopy( this, &that );
  }
  
  Object &operator =( Object const &that )
  {
    AssignCopy( this, &that );
    return *this;
  }
  
  ~Object()
  {
    Destruct( this );
  }
  
  void appendDesc( String::IOParam string ) const
  {
    if ( m_bits )
      m_bits->objectCorePtr->lTableSwapPtrPtr->get()->appendDesc( &m_bits->objectCorePtr, string );
    else string.append( "null", 4 );
  }
  
  bool isValid() const
  {
    return !!m_bits;
  }
  
  operator bool() const
  {
    return isValid();
  }
  
  bool operator !() const
  {
    return !isValid();
  }
  
  bool operator ==( INParam that )
  {
    return m_bits == that.m_bits;
  }
  
  bool operator !=( INParam that )
  {
    return m_bits != that.m_bits;
  }
  
};

template<>
struct Traits< Object >
{
  typedef Object &Result;
  typedef Object const &INParam;
  typedef Object &IOParam;
  
  static void ConstructEmpty( Object &val );
  static void ConstructCopy( Object &lhs, Object const &rhs );
  static void AssignCopy( Object &lhs, Object const &rhs );
  static void Destruct( Object &val );
};

inline void Traits<Object>::ConstructEmpty( Object &val )
{
  Object::ConstructEmpty( &val );
}
inline void Traits<Object>::ConstructCopy( Object &lhs, Object const &rhs )
{
  Object::ConstructCopy( &lhs, &rhs );
}
inline void Traits<Object>::AssignCopy( Object &lhs, Object const &rhs )
{
  Object::AssignCopy( &lhs, &rhs );
}
inline void Traits<Object>::Destruct( Object &val )
{
  Object::Destruct( &val );
}

template<>
struct Traits< MkMNIST >
{
  typedef MkMNIST &Result;
  typedef MkMNIST const &INParam;
  typedef MkMNIST &IOParam;
  
  static void ConstructEmpty( MkMNIST &val );
  static void ConstructCopy( MkMNIST &lhs, MkMNIST const &rhs );
  static void AssignCopy( MkMNIST &lhs, MkMNIST const &rhs );
  static void Destruct( MkMNIST &val );
};

}}}

#endif // __KL2EDK_AUTOGEN_global__