import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# Recommend using "HourDK" for time
power_dk1 = pd.read_xml("Electricity_Balance_Data_2011-2019_DK1.xml", parser="etree")
power_dk2 = pd.read_xml("Electricity_Balance_Data_2011-2019_DK2.xml", parser="etree")
power_dk1["HourDK"] = pd.to_datetime(power_dk1["HourDK"])
power_dk2["HourDK"] = pd.to_datetime(power_dk2["HourDK"])


# Recommend using "Date and time" for time, and "Ta (°C)" and "RH (%)" for datapoints
weather = pd.read_csv("weather.csv", decimal=",", delimiter=",")
weather["Ta (°C)"] = pd.to_numeric(weather["Ta (°C)"])
weather["RH (%)"] = pd.to_numeric(weather["RH (%)"])
weather["Date and time"] = pd.to_datetime(weather["Date and time"])


# Open image
img = np.array(Image.open("tuba.jpeg"))

# Plot image and fft of image
plt.imshow(img)
plt.savefig("figures/tuba.png")
plt.show()
spectra = np.fft.fft2(img)
plt.imshow(np.real(spectra))
plt.savefig("figures/fft_tuba.png")
plt.show()


# Focus on the power_dk1 dataset from here

# Plot a subset of the NetCon for one month
power_dk1_month = power_dk1[(power_dk1["HourDK"].dt.month == 1) & (power_dk1["HourDK"].dt.year == 2018)]
plt.plot(power_dk1_month["HourDK"], power_dk1_month["NetCon"])
plt.title("Netto power consumption DK1 January 2018")
plt.xlabel("Day of month")
plt.ylabel("Nett power consumption")
plt.savefig("figures/power_dk1_month.png")
plt.show()

# Plot the fourier transform of the same dataset
spectra = np.fft.fft(power_dk1_month["NetCon"] - np.mean(power_dk1_month["NetCon"]))
axis = np.fft.fftfreq(len(spectra))
plt.plot(axis[:(len(axis) // 2)], np.abs(spectra[:(len(axis) // 2)]))
plt.title("Fourier transform of Netto power consumption DK1 January 2018")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("figures/power_dk1_month_fourier.png")
plt.show()


# Plot the inverse fft
filtered_spectra = [x if abs(x) < 30000 else 0 for x in spectra]
other_filtered_spectra = [x if abs(x) > 30000 else 0 for x in spectra]
reconstructed = np.fft.ifft(filtered_spectra)
other_reconstructed = np.fft.ifft(other_filtered_spectra)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(power_dk1_month["HourDK"], np.real(reconstructed), color="tab:blue")
ax2.set_xlabel("Day of month")
ax1.set_ylabel("Nett power consumption")
ax2.set_ylabel("Nett power consumption")
ax2.plot(power_dk1_month["HourDK"], np.real(other_reconstructed), color="tab:orange")
fig.suptitle("Inverse Fourier transform of filtered Netto power consumption DK1 January 2018")
plt.savefig("figures/power_dk1_month_inverse_fourier.png")
plt.show()



# Repeat for the whole of 2018
power_dk1_year = power_dk1[power_dk1["HourDK"].dt.year == 2018]
plt.plot(power_dk1_year["HourDK"], power_dk1_year["NetCon"])
plt.title("Netto power consumption DK1 whole of 2018")
plt.xlabel("Time")
plt.ylabel("Nett power consumption")
plt.savefig("figures/power_dk1_year.png")
plt.show()

# Plot the fourier transform of the same dataset
spectra = np.fft.fft(power_dk1_year["NetCon"] - np.mean(power_dk1_year["NetCon"]))
axis = np.fft.fftfreq(len(spectra))
plt.plot(axis[:(len(axis) // 2)], np.abs(spectra[:(len(axis) // 2)]))
plt.title("Fourier transform of Netto power consumption DK1 whole of 2018")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("figures/power_dk1_year_fourier.png")
plt.show()


# Show the wavelet-transform of the whole year
for wavelet in ["morl", "fbsp", "gaus1"]:
    coefficients, frequencies = pywt.cwt(
        power_dk1_year["NetCon"],
        scales=np.arange(1, 500),
        wavelet=wavelet
    )

    plt.imshow(np.abs(coefficients), aspect="auto", cmap="jet", extent=[0, 365, 0, len(coefficients)])
    plt.colorbar(label="Magnitude")
    plt.ylabel("Scale")
    plt.xlabel("Day of year")
    plt.title(f"Wavelet transform of the power consumption with {wavelet}")
    plt.savefig(f"figures/power_dk1_year_wavelet_{wavelet}.png")
    plt.show()


# Look into weather data
weather_norway = weather[weather["Location"] == "Sandefjord, Norway"]
plt.plot(weather_norway["Ta (°C)"])
plt.ylabel("Temperature (°C)")
plt.xlabel("Sample number")
plt.title("Temperature in Sandefjord, Norway")
plt.savefig("figures/weather_norway.png")
plt.show()

# Plot the fourier transform of the same dataset
spectra = np.fft.fft(weather_norway["Ta (°C)"] - np.mean(weather_norway["Ta (°C)"]))
axis = np.fft.fftfreq(len(spectra))
plt.plot(axis[:(len(axis) // 2)], np.abs(spectra[:(len(axis) // 2)]))
plt.title("Fourier transform of temperature in Sandefjord, Norway")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("figures/weather_norway_fourier.png")
plt.show()

# Plot the wavelet-transform of the same dataset
for wavelet in ["morl", "fbsp", "gaus1"]:
    coefficients, frequencies = pywt.cwt(
        weather_norway["Ta (°C)"],
        scales=np.arange(1, 500),
        wavelet=wavelet
    )

    plt.imshow(np.abs(coefficients), aspect="auto", cmap="jet", extent=[0, 512, 0, len(coefficients)])
    plt.colorbar(label="Magnitude")
    plt.ylabel("Scale")
    plt.xlabel("Day of dataset")
    plt.title(f"Wavelet transform of the temperature with {wavelet}")
    plt.savefig(f"figures/weather_norway_wavelet_{wavelet}.png")
    plt.show()

# Plot moving average of the weather data
filtered = weather_norway["Ta (°C)"].rolling(1000).mean()
plt.plot(filtered)
plt.ylabel("Temperature (°C)")
plt.xlabel("Sample number")
plt.title("Moving average of temperature in Sandefjord, Norway")
plt.savefig("figures/weather_norway_filtered.png")
plt.show()
