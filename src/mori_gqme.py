from __future__ import print_function, absolute_import, division
import _mori_gqme
import f90wrap.runtime
import logging
import numpy

class Mod_Mori_Gqme(f90wrap.runtime.FortranModule):
    """
    Module mod_mori_gqme
    
    
    Defined at mod_mori_gqme.f90 lines 10-197
    
    """
    @staticmethod
    def init_module(t_mem_, k_mem_, omega_, dim_, nmem_, tf_):
        """
        init_module(t_mem_, k_mem_, omega_, dim_, nmem_, tf_)
        
        
        Defined at mod_mori_gqme.f90 lines 40-92
        
        Parameters
        ----------
        t_mem_ : float array
        k_mem_ : complex array
        omega_ : complex array
        dim_ : int
        nmem_ : int
        tf_ : float
        
        """
        if isinstance(t_mem_,(numpy.ndarray, numpy.generic)):
            if t_mem_.dtype.num != 12:
                raise TypeError
        else:
            raise TypeError
        if isinstance(k_mem_,(numpy.ndarray, numpy.generic)):
            if k_mem_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        if isinstance(omega_,(numpy.ndarray, numpy.generic)):
            if omega_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        if isinstance(dim_,(numpy.ndarray, numpy.generic)):
            if dim_.ndim != 0 or dim_.dtype.num != 5:
                raise TypeError
        elif not isinstance(dim_,int):
            raise TypeError
        if isinstance(nmem_,(numpy.ndarray, numpy.generic)):
            if nmem_.ndim != 0 or nmem_.dtype.num != 5:
                raise TypeError
        elif not isinstance(nmem_,int):
            raise TypeError
        if isinstance(tf_,(numpy.ndarray, numpy.generic)):
            if tf_.ndim != 0 or tf_.dtype.num != 12:
                raise TypeError
        elif not isinstance(tf_,float):
            raise TypeError
        _mori_gqme.f90wrap_mod_mori_gqme__init_module(t_mem_=t_mem_, k_mem_=k_mem_, \
            omega_=omega_, dim_=dim_, nmem_=nmem_, tf_=tf_)
    
    @staticmethod
    def eval_convolution(c_history_, k_mem_, dc_conv_):
        """
        eval_convolution(c_history_, k_mem_, dc_conv_)
        
        
        Defined at mod_mori_gqme.f90 lines 94-114
        
        Parameters
        ----------
        c_history_ : complex array
        k_mem_ : complex array
        dc_conv_ : complex array
        
        """
        if isinstance(c_history_,(numpy.ndarray, numpy.generic)):
            if c_history_.ndim != 3 or c_history_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        if isinstance(k_mem_,(numpy.ndarray, numpy.generic)):
            if k_mem_.ndim != 3 or k_mem_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        if isinstance(dc_conv_,(numpy.ndarray, numpy.generic)):
            if dc_conv_.ndim != 2 or dc_conv_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        _mori_gqme.f90wrap_mod_mori_gqme__eval_convolution(c_history_=c_history_, \
            k_mem_=k_mem_, dc_conv_=dc_conv_)
    
    @staticmethod
    def deriv(c_, dc_, dc_conv_):
        """
        deriv(c_, dc_, dc_conv_)
        
        
        Defined at mod_mori_gqme.f90 lines 116-121
        
        Parameters
        ----------
        c_ : complex array
        dc_ : complex array
        dc_conv_ : complex array
        
        """
        if isinstance(c_,(numpy.ndarray, numpy.generic)):
            if c_.ndim != 2 or c_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        if isinstance(dc_,(numpy.ndarray, numpy.generic)):
            if dc_.ndim != 2 or dc_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        if isinstance(dc_conv_,(numpy.ndarray, numpy.generic)):
            if dc_conv_.ndim != 2 or dc_conv_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        _mori_gqme.f90wrap_mod_mori_gqme__deriv(c_=c_, dc_=dc_, dc_conv_=dc_conv_)
    
    @staticmethod
    def rk4(istep_, c_, dt_):
        """
        rk4(istep_, c_, dt_)
        
        
        Defined at mod_mori_gqme.f90 lines 123-148
        
        Parameters
        ----------
        istep_ : int
        c_ : complex array
        dt_ : float
        
        """
        if isinstance(istep_,(numpy.ndarray, numpy.generic)):
            if istep_.ndim != 0 or istep_.dtype.num != 5:
                raise TypeError
        elif not isinstance(istep_,int):
            raise TypeError
        if isinstance(c_,(numpy.ndarray, numpy.generic)):
            if c_.ndim != 2 or c_.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        if isinstance(dt_,(numpy.ndarray, numpy.generic)):
            if dt_.ndim != 0 or dt_.dtype.num != 12:
                raise TypeError
        elif not isinstance(dt_,float):
            raise TypeError
        _mori_gqme.f90wrap_mod_mori_gqme__rk4(istep_=istep_, c_=c_, dt_=dt_)
    
    @staticmethod
    def run():
        """
        run()
        
        
        Defined at mod_mori_gqme.f90 lines 150-163
        
        
        """
        _mori_gqme.f90wrap_mod_mori_gqme__run()
    
    @staticmethod
    def finalize():
        """
        finalize()
        
        
        Defined at mod_mori_gqme.f90 lines 165-179
        
        
        """
        _mori_gqme.f90wrap_mod_mori_gqme__finalize()
    
    @staticmethod
    def euclidean_norm(x):
        """
        euclidean_norm = euclidean_norm(x)
        
        
        Defined at mod_mori_gqme.f90 lines 181-196
        
        Parameters
        ----------
        x : complex array
        
        Returns
        -------
        euclidean_norm : float
        
        """
        if isinstance(x,(numpy.ndarray, numpy.generic)):
            if x.ndim != 2 or x.dtype.num != 15:
                raise TypeError
        else:
            raise TypeError
        euclidean_norm = _mori_gqme.f90wrap_mod_mori_gqme__euclidean_norm(x=x)
        return euclidean_norm
    
    @property
    def dim(self):
        """
        Element dim ftype=integer  pytype=int
        
        
        Defined at mod_mori_gqme.f90 line 17
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__dim()
    
    @dim.setter
    def dim(self, dim):
        _mori_gqme.f90wrap_mod_mori_gqme__set__dim(dim)
    
    @property
    def nmem(self):
        """
        Element nmem ftype=integer  pytype=int
        
        
        Defined at mod_mori_gqme.f90 line 17
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__nmem()
    
    @nmem.setter
    def nmem(self, nmem):
        _mori_gqme.f90wrap_mod_mori_gqme__set__nmem(nmem)
    
    @property
    def tf(self):
        """
        Element tf ftype=real(dp) pytype=float
        
        
        Defined at mod_mori_gqme.f90 line 19
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__tf()
    
    @tf.setter
    def tf(self, tf):
        _mori_gqme.f90wrap_mod_mori_gqme__set__tf(tf)
    
    @property
    def dt(self):
        """
        Element dt ftype=real(dp) pytype=float
        
        
        Defined at mod_mori_gqme.f90 line 19
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__dt()
    
    @dt.setter
    def dt(self, dt):
        _mori_gqme.f90wrap_mod_mori_gqme__set__dt(dt)
    
    @property
    def nsteps(self):
        """
        Element nsteps ftype=integer  pytype=int
        
        
        Defined at mod_mori_gqme.f90 line 20
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__nsteps()
    
    @nsteps.setter
    def nsteps(self, nsteps):
        _mori_gqme.f90wrap_mod_mori_gqme__set__nsteps(nsteps)
    
    @property
    def t_mem(self):
        """
        Element t_mem ftype=real(dp) pytype=float
        
        
        Defined at mod_mori_gqme.f90 line 22
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__t_mem(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            t_mem = self._arrays[array_handle]
        else:
            t_mem = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__t_mem)
            self._arrays[array_handle] = t_mem
        return t_mem
    
    @t_mem.setter
    def t_mem(self, t_mem):
        self.t_mem[...] = t_mem
    
    @property
    def k_mem(self):
        """
        Element k_mem ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 23
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__k_mem(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            k_mem = self._arrays[array_handle]
        else:
            k_mem = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__k_mem)
            self._arrays[array_handle] = k_mem
        return k_mem
    
    @k_mem.setter
    def k_mem(self, k_mem):
        self.k_mem[...] = k_mem
    
    @property
    def omega(self):
        """
        Element omega ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 24
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__omega(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            omega = self._arrays[array_handle]
        else:
            omega = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__omega)
            self._arrays[array_handle] = omega
        return omega
    
    @omega.setter
    def omega(self, omega):
        self.omega[...] = omega
    
    @property
    def istep(self):
        """
        Element istep ftype=integer  pytype=int
        
        
        Defined at mod_mori_gqme.f90 line 26
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__istep()
    
    @istep.setter
    def istep(self, istep):
        _mori_gqme.f90wrap_mod_mori_gqme__set__istep(istep)
    
    @property
    def c(self):
        """
        Element c ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 27
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__c(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            c = self._arrays[array_handle]
        else:
            c = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__c)
            self._arrays[array_handle] = c
        return c
    
    @c.setter
    def c(self, c):
        self.c[...] = c
    
    @property
    def dc_conv(self):
        """
        Element dc_conv ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 29
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__dc_conv(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            dc_conv = self._arrays[array_handle]
        else:
            dc_conv = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__dc_conv)
            self._arrays[array_handle] = dc_conv
        return dc_conv
    
    @dc_conv.setter
    def dc_conv(self, dc_conv):
        self.dc_conv[...] = dc_conv
    
    @property
    def k1(self):
        """
        Element k1 ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 30
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__k1(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            k1 = self._arrays[array_handle]
        else:
            k1 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__k1)
            self._arrays[array_handle] = k1
        return k1
    
    @k1.setter
    def k1(self, k1):
        self.k1[...] = k1
    
    @property
    def k2(self):
        """
        Element k2 ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 30
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__k2(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            k2 = self._arrays[array_handle]
        else:
            k2 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__k2)
            self._arrays[array_handle] = k2
        return k2
    
    @k2.setter
    def k2(self, k2):
        self.k2[...] = k2
    
    @property
    def k3(self):
        """
        Element k3 ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 30
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__k3(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            k3 = self._arrays[array_handle]
        else:
            k3 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__k3)
            self._arrays[array_handle] = k3
        return k3
    
    @k3.setter
    def k3(self, k3):
        self.k3[...] = k3
    
    @property
    def k4(self):
        """
        Element k4 ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 30
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__k4(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            k4 = self._arrays[array_handle]
        else:
            k4 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__k4)
            self._arrays[array_handle] = k4
        return k4
    
    @k4.setter
    def k4(self, k4):
        self.k4[...] = k4
    
    @property
    def c_history(self):
        """
        Element c_history ftype=complex(dp) pytype=complex
        
        
        Defined at mod_mori_gqme.f90 line 32
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mori_gqme.f90wrap_mod_mori_gqme__array__c_history(f90wrap.runtime.empty_handle)
        if array_handle in self._arrays:
            c_history = self._arrays[array_handle]
        else:
            c_history = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mori_gqme.f90wrap_mod_mori_gqme__array__c_history)
            self._arrays[array_handle] = c_history
        return c_history
    
    @c_history.setter
    def c_history(self, c_history):
        self.c_history[...] = c_history
    
    @property
    def initialized(self):
        """
        Element initialized ftype=logical pytype=bool
        
        
        Defined at mod_mori_gqme.f90 line 34
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__initialized()
    
    @initialized.setter
    def initialized(self, initialized):
        _mori_gqme.f90wrap_mod_mori_gqme__set__initialized(initialized)
    
    @property
    def mem_tol(self):
        """
        Element mem_tol ftype=real(dp) pytype=float
        
        
        Defined at mod_mori_gqme.f90 line 37
        
        """
        return _mori_gqme.f90wrap_mod_mori_gqme__get__mem_tol()
    
    def __str__(self):
        ret = ['<mod_mori_gqme>{\n']
        ret.append('    dim : ')
        ret.append(repr(self.dim))
        ret.append(',\n    nmem : ')
        ret.append(repr(self.nmem))
        ret.append(',\n    tf : ')
        ret.append(repr(self.tf))
        ret.append(',\n    dt : ')
        ret.append(repr(self.dt))
        ret.append(',\n    nsteps : ')
        ret.append(repr(self.nsteps))
        ret.append(',\n    t_mem : ')
        ret.append(repr(self.t_mem))
        ret.append(',\n    k_mem : ')
        ret.append(repr(self.k_mem))
        ret.append(',\n    omega : ')
        ret.append(repr(self.omega))
        ret.append(',\n    istep : ')
        ret.append(repr(self.istep))
        ret.append(',\n    c : ')
        ret.append(repr(self.c))
        ret.append(',\n    dc_conv : ')
        ret.append(repr(self.dc_conv))
        ret.append(',\n    k1 : ')
        ret.append(repr(self.k1))
        ret.append(',\n    k2 : ')
        ret.append(repr(self.k2))
        ret.append(',\n    k3 : ')
        ret.append(repr(self.k3))
        ret.append(',\n    k4 : ')
        ret.append(repr(self.k4))
        ret.append(',\n    c_history : ')
        ret.append(repr(self.c_history))
        ret.append(',\n    initialized : ')
        ret.append(repr(self.initialized))
        ret.append(',\n    mem_tol : ')
        ret.append(repr(self.mem_tol))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

mod_mori_gqme = Mod_Mori_Gqme()

