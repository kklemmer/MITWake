"""
Forward marching wake TKE.

Kerry Klemmer
2024 March 12
"""

from typing import Optional 
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import interpn
from padeopsIO import key_search_r

class WakeTKE(): 
    def __init__(
            self, 
            d: float = 1,
            z_hub: float = 1, 
            xloc: float = 0,
            yloc: float = 0,
            N: int = 20, 
            sigma: float = 0.2, 
            C: float = 4., 
            lam: float = 0.214, 
            kappa: float = 0.4, 
            integrator: str = 'EF', 
            bkgd: Optional[object] = None, 
            delta_LES: Optional[object] = None,
            LES: bool = False,
            LES_prod: bool = True,
            LES_sgs_trans: bool = True,
            LES_pres_trans: bool = True,
            LES_turb_trans: bool = True,
            LES_diss: bool = True,
            LES_buoy: bool = True,
            LES_base_adv: bool = True,
            LES_delta_vw: bool = True,
            avg_xy: bool = True, 
            dx: float = 0.05, 
            dy: float = 0.1, 
            dz: float = 0.1, 
    ): 
        """
        Args: 
            d (float): non-dimensionalizing value for diameter. Defaults to 1.
            z_hub (float): non-dimensional hub height z_hub/d. Defaults to 1. 
            N (int): number of points to discretize Lamb-Oseem vortices. Defaults to 20. 
            sigma (float): width of Lamb-Oseem vortices. Defaults to 0.2. 
            C (float): mixing length constant. Defaults to 4.
            lam (float): free atmosphere mixing length. Defaults to 0.214 (27 m for the NREL 5 MW). 
            kappa (float): von Karman constant. Defaults to 0.4. 
            integrator (str): forward integration method. Defaults to 'EF'.
            bkgd (PadeOpsIO.BudgetIO): object with background flow properties. Defaults to None. 
            avg_xy (bool): average background flow in x, y. Defaults to True. 
        """

        self.d = d  # TODO - be more consistent in the dimensional problem
        self.z_hub = z_hub
        self.xloc = xloc
        self.yloc = yloc
        self.N = N
        self.sigma = sigma
        self.C = C
        self.lam = lam
        self.kappa = kappa
        self.integrator = get_integrator(integrator)
        
        self.bkgd = bkgd  # TODO - fix this down the line
        self.delta_LES = delta_LES
        self.LES = LES
        self.LES_prod = LES_prod
        self.LES_sgs_trans = LES_sgs_trans
        self.LES_pres_trans = LES_pres_trans
        self.LES_turb_trans = LES_turb_trans
        self.LES_diss = LES_diss
        self.LES_buoy = LES_buoy
        self.LES_base_adv = LES_base_adv
        self.LES_delta_vw = LES_delta_vw
        self.avg_xy = avg_xy
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.has_result = False  # true when solved for the wake solution
        self.extent = None

        # initialize background flow and grid
        self.init_grid()
        self.init_background()
        if delta_LES is not None:
            self.init_delta_uvw()
        if self.LES:
            self.init_delta_LES()
        
    def compute_tke(
            self, 
            x: np.array, 
            y: np.array, 
            z: np.array, 
            prim: bool = False,
            return_tke: bool = False, 
    )-> None: 
        """
        Computes the wake deficit tke field via forward marching. 
        """        
        self.init_grid(x, y, z)  # creates a (possibly new) grid

        if not prim:
            self._interp_Ui()
            self._interp_delta_ui()
            if self.LES:
                self._interp_delta_LES()
                # if self.smooth_LES:
                #     self._smooth_LES()
            elif self.base_gradients:
                self._interp_Ui_3D()

            self._compute_uvw()
            # compute deficit field
            self._compute_tke() 
        else:
            self._interp_prim_LES()

            self._compute_prim_tke()

        self.has_result = True

        if return_tke: 
            return (self.tke)

    def init_grid(self, x=0, y=0, z=0, yz_buff: float = .6,): 
        """
        Creates a grid centered around the wind turbine. 
        """

        def _make_axis(xmin, xmax, dx): 
            """Helper function to create an axis from limits that always includes 0"""
            n_st = np.ceil(np.abs(xmin / dx)) * np.sign(xmin)
            n_end = np.ceil(np.abs(xmax / dx)) * np.sign(xmax)
            return np.arange(n_st, n_end + dx) * dx  # ensure that 0 is included in the grid
        
        # if self.LES:

        #     self.xg = self.delta_LES.xLine
        #     self.yg = self.delta_LES.yLine
        #     self.zg = self.delta_LES.zLine

        #     self.dx = self.delta_LES.dx
        #     self.dy = self.delta_LES.dy
        #     self.dz = self.delta_LES.dz

        # else:
        # set limits with buffers: 
        xlim = [np.min([-self.dx, np.min(x)]), np.max(x)]
        ylim = [np.min([-yz_buff, np.min(y)]), np.max([yz_buff, np.max(y)])]
        zlim = [np.min([-yz_buff, np.min(z)]), np.max([yz_buff, np.max(z)])]
        #if self.bkgd is not None:
        #    zlim[0] = max(zlim[0],-self.z_hub)

        self.xg = _make_axis(*xlim, self.dx)
        self.yg = _make_axis(*ylim, self.dy)
        self.zg = _make_axis(*zlim, self.dz)

        self.shape = (len(self.xg), len(self.yg), len(self.zg))

        self.extent = [min(self.xg), max(self.xg), 
                       min(self.yg), max(self.yg), 
                       min(self.zg), max(self.zg)]  # these bounds may differ from inputs
        
    def within_bounds(self, x, y, z): 
        """Check if new axes are within computed bounds"""
        for xi, Xi in zip([x, y, z], [self.xg, self.yg, self.zg]): 
            if np.min(xi) < np.min(Xi) or np.max(xi) > np.max(Xi): 
                return False
            
        return True

    def init_background(self): 
        """
        Initialize background flow fields. 
        """
        if self.bkgd is not None: 
            # LES background flow
            self.bkgd.read_budgets(budget_terms=['ubar', 'vbar', 'wbar']) 
            Ug = key_search_r(self.bkgd.input_nml, 'g_geostrophic')
            if self.avg_xy: 
                Ui = self.bkgd.budget
                self._U = np.mean(Ui['ubar'], (0, 1))/Ug
                self._V = np.mean(Ui['vbar'], (0, 1))/Ug
                try: 
                    self._W = np.mean(Ui['wbar'], (0, 1))/Ug
                except KeyError as e: 
                    self._W = np.zeros_like(self._V)  # should have zero subsidence
                self._z = self.bkgd.zLine - self.z_hub
                self._interp_Ui()

            else: 
                raise NotImplementedError('init_background(): Currently `avg_xy` must be True.')
            if self.LES:
                self._U_3D = self.bkgd.budget['ubar']/Ug
                self._x = self.bkgd.xLine - self.xloc
                self._y = self.bkgd.yLine - self.yloc
                self._interp_Ui_3D()
        else: 
            self.U = np.ones_like(self.zg)  # uniform inflow
            self.V = 0 
            self.W = 0

    def init_delta_uvw(self):
        """
        Initializes delta v and delta w from data
        """
        if self.delta_LES is not None:
            self.delta_LES.read_budgets(budget_terms=['delta_u','delta_v', 'delta_w'])
            Ug = key_search_r(self.delta_LES.input_nml, 'g_geostrophic')
            # if self.avg_xy: 
            delta_ui = self.delta_LES.budget
            self._delta_u = delta_ui['delta_u']/Ug
            self._delta_v = delta_ui['delta_v']/Ug
            self._delta_w = delta_ui['delta_w']/Ug
            self._delta_x = self.delta_LES.xLine
            self._delta_y = self.delta_LES.yLine
            self._delta_z = self.delta_LES.zLine
            self._interp_delta_ui()

            # smooth data

    def init_delta_LES(self):
        """
        Initializes RHS of wake TKE transport from LES data
        """
        Ug = key_search_r(self.delta_LES.input_nml, 'g_geostrophic')
        Lref = self.delta_LES.Lref
        D = self.delta_LES.D

        nonDim = Ug*Ug*Ug*Lref/D

        delta_budget = self.delta_LES.budget
        self._tke_LES = delta_budget['TKE_wake']/(Ug*Ug)
        self._pres_trans = delta_budget['TKE_p_transport_wake']/nonDim
        self._sgs_trans = delta_budget['TKE_SGS_transport_wake']/nonDim
        self._turb_trans = delta_budget['TKE_turb_transport_wake']/nonDim
        self._prod = delta_budget['TKE_shear_production_wake']/nonDim
        self._buoy = delta_budget['TKE_buoyancy_wake']/nonDim
        self._diss = delta_budget['TKE_dissipation_wake']/nonDim
        self._base_adv = delta_budget['TKE_adv_delta_base_k_wake']/nonDim

    def init_prim_LES(self):
        """
        Initializes RHS of primary TKE transport from LES data
        """
        Ug = key_search_r(self.delta_LES.input_nml, 'g_geostrophic')
        Lref = self.delta_LES.Lref
        D = self.delta_LES.D

        nonDim = Ug*Ug*Ug*Lref/D

        prim_budget = self.prim_LES.budget
        self._tke_LES = prim_budget['TKE']/(Ug*Ug)
        self._pres_trans = prim_budget['TKE_p_transport']/nonDim
        self._sgs_trans = prim_budget['TKE_SGS_transport']/nonDim
        self._turb_trans = prim_budget['TKE_turb_transport']/nonDim
        self._prod = prim_budget['TKE_shear_production']/nonDim
        self._buoy = prim_budget['TKE_buoyancy']/nonDim
        self._diss = prim_budget['TKE_dissipation']/nonDim
        self._base_adv = prim_budget['TKE_adv_delta_base_k_wake']/nonDim

    def _interp_Ui(self)-> None: 
        """Interpolate velocity profiles to local grid"""
        if self.bkgd is not None: 
            self.U = np.interp(self.zg, self._z, self._U)
            self.V = np.interp(self.zg, self._z, self._V)
            self.W = np.interp(self.zg, self._z, self._W)
        else: 
            self.U = np.ones_like(self.zg)

    def _interp_Ui_3D(self)-> None: 
        """Interpolate velocity profiles to local grid"""
         
        xG, yG, zG = np.meshgrid(self.xg, self.yg, self.zg, indexing='ij')
        xl = np.ravel(xG)
        yl = np.ravel(yG)
        zl = np.ravel(zG)

        interp_points = np.stack([xl, yl, zl], axis=-1)

        self.U_3D = interpn((self._x, self._y, self._z), self._U_3D, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

    def _interp_delta_LES(self)-> None: 
        """Interpolate velocity profiles to local grid"""
        
        xG, yG, zG = np.meshgrid(self.xg, self.yg, self.zg, indexing='ij')
        xl = np.ravel(xG)
        yl = np.ravel(yG)
        zl = np.ravel(zG)

        interp_points = np.stack([xl, yl, zl], axis=-1)

        self.tke_LES = interpn((self._delta_x, self._delta_y, self._delta_z), self._tke_LES, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

        self.pres_trans = interpn((self._delta_x, self._delta_y, self._delta_z), self._pres_trans, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

        self.sgs_trans = interpn((self._delta_x, self._delta_y, self._delta_z), self._sgs_trans, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

        self.turb_trans = interpn((self._delta_x, self._delta_y, self._delta_z), self._turb_trans, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

        self.prod = interpn((self._delta_x, self._delta_y, self._delta_z), self._prod, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

        self.buoy = interpn((self._delta_x, self._delta_y, self._delta_z), self._buoy, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

        self.diss = interpn((self._delta_x, self._delta_y, self._delta_z), self._diss, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

        self.base_adv = interpn((self._delta_x, self._delta_y, self._delta_z), self._base_adv, interp_points, 
                bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

    def _interp_delta_ui(self)-> None: 
        """Interpolate velocity profiles to local grid"""
        if self.delta_LES is not None: 
            xG, yG, zG = np.meshgrid(self.xg, self.yg, self.zg, indexing='ij')
            xl = np.ravel(xG)
            yl = np.ravel(yG)
            zl = np.ravel(zG)

            interp_points = np.stack([xl, yl, zl], axis=-1)
            
            self.delta_u = interpn((self._delta_x, self._delta_y, self._z), self._delta_u, interp_points, bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg))) 
            self.delta_v = interpn((self._delta_x, self._delta_y, self._z), self._delta_v, interp_points, bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))
            self.delta_w = interpn((self._delta_x, self._delta_y, self._z), self._delta_w, interp_points, bounds_error=False, fill_value=0).reshape((len(self.xg), len(self.yg), len(self.zg)))

    def _smooth_LES(self)-> None:
        """Smooth LES data"""
        # if self.smooth_LES:
        #     for i in range(len(self.xg)):
        #         self.u_LES[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.u_LES[i,...])
        #         self.delta_v[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.delta_v[i,...])
        #         self.delta_w[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.delta_w[i,...])
        #         self.dpdx[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.dpdx[i,...])
        #         self.RS_div[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.RS_div[i,...])
        #         self.sgs[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.sgs[i,...])
        #         self.xCor[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.xCor[i,...])
        #         self.base_grad[i,...] = np.apply_along_axis(lambda m: self.moving_avg(m), axis=0, arr=self.base_grad[i,...])


    def get_ic(
            self, 
            y: np.array, 
            z: np.array, 
            ud: float, 
            smooth_fact: float = 1.5, 
    )-> np.array: 
        """
        Initial condition for the wake model

        Args: 
            self (CurledWake)
            y (np.array): lateral axis
            z (np.array): vertical axis
            ud (float): disk velocity
            smooth_fact (float): Gaussian convolution standard deviation, 
                equal to smooth_fact * self.dy. Defaults to 1.5. 
        """
        yG, zG = np.meshgrid(y, z, indexing='ij')
        kernel_y = np.arange(-10, 11)[:, None] * self.dy
        kernel_z = np.arange(-10, 11)[None, :] * self.dz

        turb = (yG**2 + zG**2) < (self.d / 2)**2
        # gauss = np.exp(-(yG**2 + zG**2) / (np.sqrt(self.dy * self.dz) * smooth_fact)**2 / 2)
        gauss = np.exp(-(kernel_y**2 + kernel_z**2) / (np.sqrt(self.dy * self.dz) * smooth_fact)**2 / 2)
        gauss /= np.sum(gauss)
        a = 0.5 * (1 - np.sqrt(1 - self.ct * np.cos(self.yaw)**2))
        print(a)
        delta_u = -2 * a * ud

        return convolve2d(turb, gauss, 'same') * delta_u
    
    def get_ud(self, weighting='disk')-> float: 
        """
        Gets background disk velocity by numerically integrating self.U
        """
        if self.U.ndim == 1: 
            r = self.d/2
            zids = abs(self.zg) < r  # within the rotor area
            if weighting == 'disk': 
                # weight based on the area of the "disk"
                A = np.trapz(np.sqrt(r**2 - self.zg[zids]**2)) 
                return np.trapz(self.U[zids] * np.sqrt(r**2 - self.zg[zids]**2)) / A
            
            elif weighting == 'equal': 
                return np.mean(self.U[zids])
            elif weighting == 'hub': 
                return np.interp(0, self.zg, self.U)  # hub height velocity
            else: 
                raise NotImplementedError('get_ud(): `weighting` must be "disk", "equal", or "hub". ')
        else: 
            # TODO - FIX
            raise NotImplementedError('Deal with this later')

    def get_virtual_ud(self, xloc)-> float: 
        """
        Gets disk velocity of a virtual turbine by numerically integrating self.u + self.U
        """
        xid, yid, zid, = xid, yid, zid, = self.get_xids(x=xloc,y=[-0.5,0.5],z=[-0.5,0.5], return_none=True, return_slice=True)

        self.kernel = np.zeros([len(self.yg), len(self.zg)])
        self.kernel[yid,zid] = 1
        self.kernel /= np.sum(self.kernel)

        ud = np.sum(np.multiply(self.kernel,np.add(self.u[xid,...], self.U)))

        return ud

    def _compute_tke(self, ud=None)-> None: 
        """
        Forward marches Delta_u field. 
        """
        def _dkdt(x, _tke): 
            """du/dt function"""
            xid = np.argmin(np.abs(x - self.xg))
            # Full velocity fields for advection: 
            u = self.u[xid, ...] + self.U
            v = self.v[xid, ...] + self.V
            w = self.w[xid, ...] + self.W 
            # gradient fields: 
            dkdy = np.gradient(_tke, self.dy, axis=0)
            dkdz = np.gradient(_tke, self.dz, axis=1)
            d2kdy2 = np.gradient(dkdy, self.dy, axis=0)
            d2kdz2 = np.gradient(dkdz, self.dz, axis=1)
            # base gradient fields:   
            rhs_value = 0         
            if self.LES:
                if self.LES_pres_trans:
                    rhs_value += self.pres_trans[xid,...]
                if self.LES_sgs_trans:
                    rhs_value += self.sgs_trans[xid,...]
                if self.LES_turb_trans:
                    rhs_value += self.turb_trans[xid,...]
                if self.LES_prod:
                    rhs_value += self.prod[xid,...]
                if self.LES_buoy:
                    rhs_value += self.buoy[xid,...]
                if self.LES_diss:
                    rhs_value += self.diss[xid,...]
                if self.LES_base_adv:
                    rhs_value += self.base_adv[xid,...]
        
            return (-v * dkdy - w * dkdz + rhs_value) / u
            
        # now integrate! 
        if ud is None:
            ud = self.get_ud()

        if self.LES:
            # xval = np.argmin(abs(self.xg))
            # ic = self.xAD[xval,...]
            xval = np.argmin(abs(self.xg-5))
            ic = self.tke_LES[xval,...]
            xmin = 5
            self.ic = ic
        else:
            ic = self.get_ic(self.yg, self.zg, ud)
            xmin = 0 
        # xmin = 0
        xmax = max(self.xg)
        x, tke = integrate(ic, _dkdt, dt=self.dx, T=[xmin, xmax], f=self.integrator)

        self.tke = np.zeros(self.shape)
        xid_st = np.argmin(abs(self.xg-xmin))
        self.tke[xid_st:, ...] = tke

    def _compute_uvw(self)-> None: 
        """
        Use Lamb-Oseen vortices to compute curling
        """

        self.u = self.delta_u

        if self.LES and self.LES_delta_vw:
            self.v = self.delta_v
            self.w = self.delta_w
            return
        else: 
            self.v = np.zeros(self.shape)
            self.w = np.zeros(self.shape)
            return
        u_inf = self.get_ud('hub')
        r_i = np.linspace(-(self.d - self.dz) / 2, (self.d - self.dz) / 2, self.N) 

        Gamma_0 = 0.5 * self.d * u_inf * self.ct * np.sin(self.yaw) * np.cos(self.yaw)**2
        Gamma_i = Gamma_0 * 4 * r_i / (self.N * self.d**2 * np.sqrt(1 - (2 * r_i / self.d)**2))

        # generally, vortices can decay. So sigma should be a vector
        sigma = self.sigma * self.d * np.ones_like(self.xg) 
        
        # now we build the main summation, which is 4D (x, y, z, i)
        # yg_total = np.arange(self.yg[0]-self.yg[-1], self.yg[-1]+self.dy, self.dy)
        # zg_total = np.arange(self.zg[0]-self.zg[-1], self.zg[-1]+self.dz, self.dz)
        # yG, zG = np.meshgrid(yg_total, zg_total, indexing='ij')
        yG, zG = np.meshgrid(self.yg, self.zg, indexing='ij')
        yG = yG[None, ..., None]
        zG = zG[None, ..., None]
        r4D = yG**2 + (zG - r_i[None, None, None, :])**2  # 4D grid variable

        # mask for the ground effect
        # ind_y = np.argmin(np.abs(yg_total - self.yg[0]))
        # ind_z = np.argmin(np.abs(zg_total - self.zg[0]))
        # mask = np.ones(np.shape(r4D))
        # mask[:,:,:ind_y,:ind_z] = -1

        # put pieces together: 
        exponent = 1 - np.exp(-r4D / sigma[..., None, None, None]**2)
        summation = exponent / (2 * np.pi * r4D) * Gamma_i[None, None, None, :]

        v = np.sum(summation * (zG - r_i[None, None, None, :]), axis=-1)  # sum all vortices
        w = np.sum(summation * -yG, axis=-1)
        self.v = v * (self.xg >= 0)[:, None, None]
        self.w = w * (self.xg >= 0)[:, None, None]

    def get_rmse(self, xlim=None, ylim=None, zlim=None):
        """
        Calculates the rmse from self.error
        self.error should be the difference between self.u and self.delta_u_LES
        and needs to already be defined

        Parameters
        ----------
        xlim, ylim, zlim : (tuple)
            in physical domain coordinates, the slice limits. If an integer is given, then the 
            dimension of the slice will be reduced by one. If None is given (default), then the entire domain extent is sliced.  

        Returns
        -------
        rmse : float
            RMSE 
        """

        xid, yid, zid = self.get_xids(x=xlim,y=ylim,z=zlim, return_none=True, return_slice=True)

        return np.sqrt(np.mean(np.abs(np.power(self.error[xid,yid,zid],2))))

    def moving_avg(self, signal, kernel_size=10): 
        kernel = np.ones(kernel_size)
        normfact = np.convolve(np.zeros_like(signal) + 1, kernel, mode='same')
        smoothed = np.convolve(signal, kernel, mode='same')

        return smoothed/normfact

    def get_l2norm(self, xlim=None, ylim=None, zlim=None):
        """                                                                                                                                                                       
        Calculates the rmse from self.error                                                                                                                                       
        self.error should be the difference between self.u and self.delta_u_LES                                                                                                   
        and needs to already be defined                                                                                                                                           
                                                                                                                                                                                  
        Parameters                                                                                                                                                                
        ----------                                                                                                                                                                
        xlim, ylim, zlim : (tuple)                                                                                                                                                
            in physical domain coordinates, the slice limits. If an integer is given, then the                                                                                    
            dimension of the slice will be reduced by one. If None is given (default), then the entire domain extent is sliced.                                                   
                                                                                                                                                                                  
        Returns                                                                                                                                                                   
        -------                                                                                                                                                                   
        rmse : float                                                                                                                                                              
            RMSE                                                                                                                                                                  
        """

        xid, yid, zid = self.get_xids(x=xlim,y=ylim,z=zlim, return_none=True, return_slice=True)

        return np.sqrt(np.sum(np.abs(np.power(self.error[xid,yid,zid],2))))

    def get_xids(self, **kwargs): 
        """
        Translates x, y, and z limits in the physical domain to indices based on self.xLine, self.yLine, and self.zLine

        Arguments
        ---------
        x, y, z : float or iterable (tuple, list, etc.) of physical locations to return the nearest index for

        Returns
        -------
        xid, yid, zid : list or tuple of lists with indices for the requested x, y, z, args in the order: x, y, z. 
            If, for example, y and z are requested, then the returned tuple will have (yid, zid) lists. 
            If only one value (float or int) is passed in for e.g. x, then an integer will be passed back in xid. 
        """

        # set up this way in case we want to introduce an offset later on (i.e. turbine-centered coordinates)
        if 'x_ax' not in kwargs or kwargs['x_ax'] is None: 
            kwargs['x_ax'] = self.xg 
        if 'y_ax' not in kwargs or kwargs['y_ax'] is None: 
            kwargs['y_ax'] = self.yg  
        if 'z_ax' not in kwargs or kwargs['z_ax'] is None: 
            kwargs['z_ax'] = self.zg 

        return get_xids(**kwargs)


def get_integrator(integrator): 
    """Return integrator function"""
    if integrator == 'EF': 
        return EF_step
    elif integrator == 'RK4': 
        return rk4_step
    else: 
        raise ValueError(f'"{integrator}" not a valid integration function')


def rk4_step(t_n, u_n, dudt, dt): 
    """
    Computes the next timestep of u_n given the finite difference function du/dt
    with a 4-stage, 4th order accurate Runge-Kutta method. 
    
    Parameters
    ----------
    t_n : float
        time for time step n
    u_n : array-like
        condition at time step n
    dudt : function 
        function du/dt(t, u)
    dt : float
        time step
    
    Returns u_(n+1)
    """    
    k1 = dt * dudt(t_n, u_n)
    k2 = dt * dudt(t_n + dt/2, u_n + k1/2)
    k3 = dt * dudt(t_n + dt/2, u_n + k2/2)
    k4 = dt * dudt(t_n + dt, u_n + k3)

    u_n1 = u_n + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u_n1


def EF_step(t_n, u_n, dudt, dt): 
    """
    Forward Euler stepping scheme
    """
    u_n1 = u_n + dt * dudt(t_n, u_n)
    return u_n1


def integrate(u0, dudt, dt=0.005, T=[0, 1], f=rk4_step): 
    """
    General integration function which calls a step function multiple times depending 
    on the parabolic integration strategy. 

    Parameters
    ----------
    u0 : array-like
        Initial condition of values
    dudt : function 
        Evolution function du/dt(t, u, ...)
    dt : float
        Time step
    T : (2, ) 
        Time range
    f : function
        Integration stepper function (e.g. RK4, EF, etc.)

    Returns
    -------
    t : (Nt, ) vector
        Time vector 
    u(t) : (Nt, ...) array-like
        Solution to the parabolic ODE. 
    """
    t = []
    ut = []

    u_n = u0  # initial condition
    t_n = T[0]
    
    while True: 
        ut.append(u_n)
        t.append(t_n)

        # update timestep
        t_n1 = t_n + dt
        if t_n1 > T[1] + dt/2:  # add some buffer here
            break
        u_n1 = f(t_n, u_n, dudt, dt)

        # update: 
        u_n = u_n1
        t_n = t_n1

    return np.array(t), np.array(ut)


def get_xids(x=None, y=None, z=None, 
             x_ax=None, y_ax=None, z_ax=None, 
             return_none=False, return_slice=False): 
    """
        Translates x, y, and z limits in the physical domain to indices based on self.xLine, self.yLine, and self.zLine

        Parameters
        ---------
        x, y, z : float or iterable (tuple, list, etc.) 
            Physical locations to return the nearest index 
        return_none : bool
            If True, populates output tuple with None if input is None. Default False. 
        return_slice : bool 
            If True, returns a tuple of slices instead a tuple of lists. Default False. 

        Returns
        -------
        xid, yid, zid : list or tuple of lists 
            Indices for the requested x, y, z, args in the order: x, y, z. 
            If, for example, y and z are requested, then the returned tuple will have (yid, zid) lists. 
            If only one value (float or int) is passed in for e.g. x, then an integer will be passed back in xid. 
    """

    ret = ()

    # iterate through x, y, z, index matching for each term
    for s, s_ax in zip([x, y, z], [x_ax, y_ax, z_ax]): 
        if s is not None: 
            if s_ax is None: 
                raise AttributeError('Axis keyword not provided')
                
            if hasattr(s, '__iter__'): 
                xids = [np.argmin(np.abs(s_ax-xval)) for xval in s]
            else: 
                xids = np.argmin(np.abs(s_ax-s))

            xids = np.squeeze(np.unique(xids))

            if return_slice and xids.ndim > 0:  # append slices to the return tuple
                ret = ret + (slice(np.min(xids), np.max(xids)+1), )

            else:  # append index list to the return tuple
                ret = ret + (xids, )

        elif return_none:  # fill with None or slice(None)
            if return_slice: 
                ret = ret + (slice(None), )

            else: 
                ret = ret + (None, )

    if len(ret)==1: 
        return ret[0]  # don't return a length one tuple 
    else: 
        return ret

