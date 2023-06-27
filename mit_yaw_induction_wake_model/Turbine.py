# typing.Literal was introduced in Python3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from mit_yaw_induction_wake_model import Rotor, Wake
import numpy as np


class BasicTurbine:
    def __init__(
        self,
        Ct: float,
        yaw: float,
        x=0.0,
        y=0.0,
        sigma=0.25,
        kw=0.07,
        induction_eps=0.000001,
    ) -> None:
        """

        Args:
            Ct (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).
            x (float): Longitudinal turbine position. Defaults to 0.0.
            y (float): Lateral turbine position. Defaults to 0.0.
            sigma (float): Gaussian wake proportionality constant. Defaults to 0.25.
            kw (float): Wake spreading parameter. Defaults to 0.07.
            induction_eps (float): Convergence tolerance. Defaults to 0.000001.
        """
        self.x, self.y = x, y
        self.Ct, self.yaw = Ct, yaw
        self.a, u4, v4 = Rotor.yawthrust(Ct, yaw, eps=induction_eps)
        self.wake = Wake.Gaussian(u4, v4, sigma, kw)

    def deficit(
        self, x: np.ndarray, y: np.ndarray, z=0, FOR: Literal["met", "local"] = "met"
    ) -> np.ndarray:
        """
        Returns the wake deficit generated by the turbine sampled at the given
        positions in space.

        Args:
            x (np.ndarray): Longitudinal positions to sample.
            y (np.ndarray): Lateral positions to sample.
            z (int): Vertical positions to sample. Defaults to 0.
            FOR (Literal["met", "local"]): Frame of reference. Defaults to
            "met". "met" indicates the farm coordinate system where the
            freestream velocity is along the x axis. "local" indicates the
            turbine frame of reference where (0, 0) is the location of the
            turbine.

        Returns:
            np.ndarray: Wake deficit at sample points.
        """
        if FOR == "met":
            return self.wake.deficit(x - self.x, y - self.y, z)
        elif FOR == "local":
            return self.wake.deficit(x, y, z)


class GradientTurbine:
    def __init__(
        self,
        Ct: float,
        yaw: float,
        x=0.0,
        y=0.0,
        sigma=0.25,
        kw=0.07,
        induction_eps=0.000001,
    ):
        """
        Args:
            Ct (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).
            x (float): Longitudinal turbine position. Defaults to 0.0.
            y (float): Lateral turbine position. Defaults to 0.0.
            sigma (float): Gaussian wake proportionality constant. Defaults to 0.25.
            kw (float): Wake spreading parameter. Defaults to 0.07.
            induction_eps (float): Convergence tolerance. Defaults to 0.000001.
        """
        self.x, self.y = x, y
        self.Ct, self.yaw = Ct, yaw
        (
            self.a,
            u4,
            v4,
            self.dadCt,
            dudCt,
            dvdCt,
            self.dadyaw,
            dudyaw,
            dvdyaw,
        ) = Rotor.gradyawthrust(Ct, yaw, eps=induction_eps)
        self.wake = Wake.GradGaussian(u4, v4, dudCt, dudyaw, dvdCt, dvdyaw, sigma, kw)

    def deficit(
        self, x: np.ndarray, y: np.ndarray, z=0, FOR: Literal["met", "local"] = "met"
    ) -> np.ndarray:
        """
        Returns the wake deficit generated by the turbine sampled at the given
        positions in space.

        Args:
            x (np.ndarray): Longitudinal positions to sample.
            y (np.ndarray): Lateral positions to sample.
            z (int): Vertical positions to sample. Defaults to 0.
            FOR (Literal["met", "local"]): Frame of reference. Defaults to
            "met". "met" indicates the farm coordinate system where the
            freestream velocity is along the x axis. "local" indicates the
            turbine frame of reference where (0, 0) is the location of the
            turbine.

        Returns:
            np.ndarray: Wake deficit at sample points.
        """
        if FOR == "met":
            return self.wake.deficit(x - self.x, y - self.y, z)
        elif FOR == "local":
            return self.wake.deficit(x, y, z)
