/*
 * Evaluate OSKAR aperture-array Jones matrices and contract only the station
 * pairs needed by a visibility sample. The output is
 * 0.5 * Tr(E_p E_q^H), ordered as [row, source].
 */

#include <apps/oskar_app_settings.h>
#include <apps/oskar_settings_to_telescope.h>
#include <gains/oskar_gains.h>
#include <interferometer/oskar_evaluate_jones_E.h>
#include <interferometer/oskar_jones.h>
#include <log/oskar_log.h>
#include <mem/oskar_mem.h>
#include <settings/oskar_SettingsTree.h>
#include <telescope/oskar_telescope.h>
#include <telescope/station/oskar_station_work.h>
#include <utility/oskar_get_error_string.h>
#include <utility/oskar_vector_types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string config;
    std::string directions;
    std::string rows;
    std::string output;
    std::int64_t num_sources = 0;
    std::int64_t num_rows = 0;
    std::int64_t source_chunk = 32768;
};

struct Row {
    std::int32_t time_index;
    std::int32_t antenna1;
    std::int32_t antenna2;
};

void check_status(int status, const std::string& context)
{
    if (status)
    {
        throw std::runtime_error(
                context + ": " + oskar_get_error_string(status));
    }
}

std::int64_t parse_positive(const char* value, const char* name)
{
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (!end || *end != '\0' || parsed < 1)
    {
        throw std::runtime_error(std::string("Invalid ") + name);
    }
    return static_cast<std::int64_t>(parsed);
}

Arguments parse_arguments(int argc, char** argv)
{
    Arguments args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key(argv[i]);
        if (i + 1 >= argc)
        {
            throw std::runtime_error("Missing value after " + key);
        }
        const char* value = argv[++i];
        if (key == "--config") args.config = value;
        else if (key == "--directions") args.directions = value;
        else if (key == "--rows") args.rows = value;
        else if (key == "--output") args.output = value;
        else if (key == "--num-sources")
            args.num_sources = parse_positive(value, "num-sources");
        else if (key == "--num-rows")
            args.num_rows = parse_positive(value, "num-rows");
        else if (key == "--source-chunk")
            args.source_chunk = parse_positive(value, "source-chunk");
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (args.config.empty() || args.directions.empty() ||
            args.rows.empty() || args.output.empty() ||
            args.num_sources < 1 || args.num_rows < 1)
    {
        throw std::runtime_error("Required arguments are missing");
    }
    return args;
}

template <typename T>
std::vector<T> read_binary(const std::string& path, std::size_t count)
{
    std::vector<T> values(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Cannot open input: " + path);
    }
    stream.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof())
    {
        throw std::runtime_error("Unexpected input byte count: " + path);
    }
    return values;
}

std::complex<double> as_complex(const double2& value)
{
    return {value.x, value.y};
}

std::complex<float> stokes_i_factor(
        const double4c& first, const double4c& second)
{
    const std::complex<double> value = 0.5 * (
            as_complex(first.a) * std::conj(as_complex(second.a)) +
            as_complex(first.b) * std::conj(as_complex(second.b)) +
            as_complex(first.c) * std::conj(as_complex(second.c)) +
            as_complex(first.d) * std::conj(as_complex(second.d)));
    return {
        static_cast<float>(value.real()),
        static_cast<float>(value.imag())
    };
}

void initialise_output(
        std::fstream& output, const std::string& path, std::uint64_t bytes)
{
    output.open(
            path,
            std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!output || bytes < 1)
    {
        throw std::runtime_error("Cannot create output: " + path);
    }
    output.seekp(static_cast<std::streamoff>(bytes - 1));
    const char zero = 0;
    output.write(&zero, 1);
    output.flush();
    if (!output)
    {
        throw std::runtime_error("Cannot size output: " + path);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    oskar::SettingsTree* settings = nullptr;
    oskar_Log* log = nullptr;
    oskar_Telescope* telescope_cpu = nullptr;
    oskar_Telescope* telescope_gpu = nullptr;
    oskar_StationWork* work = nullptr;
    oskar_Jones* jones = nullptr;
    oskar_Mem* station_cpu = nullptr;
    int status = 0;
    try
    {
        const Arguments args = parse_arguments(argc, argv);
        const auto started = std::chrono::steady_clock::now();
        const std::vector<double> directions = read_binary<double>(
                args.directions,
                static_cast<std::size_t>(3 * args.num_sources));
        const std::vector<std::int32_t> row_values =
                read_binary<std::int32_t>(
                        args.rows,
                        static_cast<std::size_t>(3 * args.num_rows));
        std::vector<Row> rows(static_cast<std::size_t>(args.num_rows));
        for (std::int64_t i = 0; i < args.num_rows; ++i)
        {
            rows[static_cast<std::size_t>(i)] = {
                row_values[static_cast<std::size_t>(3 * i)],
                row_values[static_cast<std::size_t>(3 * i + 1)],
                row_values[static_cast<std::size_t>(3 * i + 2)]
            };
        }

        settings = oskar_app_settings_tree(
                "oskar_sim_interferometer", args.config.c_str());
        if (!settings)
        {
            throw std::runtime_error("Cannot load OSKAR settings");
        }
        settings->clear_group();
        settings->begin_group("observation");
        const double start_mjd = settings->to_double(
                "start_time_utc", &status);
        const double length_sec = settings->to_double("length", &status);
        const int num_times = settings->to_int("num_time_steps", &status);
        const double frequency_hz = settings->to_double(
                "start_frequency_hz", &status);
        settings->end_group();
        check_status(status, "Read observation settings");
        if (num_times < 1 || !std::isfinite(start_mjd) ||
                !std::isfinite(length_sec) || length_sec <= 0.0 ||
                !std::isfinite(frequency_hz) || frequency_hz <= 0.0)
        {
            throw std::runtime_error("Invalid observation settings");
        }

        log = oskar_log_create(OSKAR_LOG_NONE, OSKAR_LOG_NONE);
        telescope_cpu = oskar_settings_to_telescope(settings, log, &status);
        check_status(status, "Load telescope");
        oskar_telescope_analyse(telescope_cpu, &status);
        check_status(status, "Analyse telescope");
        if (!oskar_telescope_allow_station_beam_duplication(telescope_cpu))
        {
            throw std::runtime_error(
                    "Exact row-beam contraction currently requires "
                    "allow_station_beam_duplication=true");
        }
        if (oskar_gains_defined(oskar_telescope_gains_const(telescope_cpu)))
        {
            throw std::runtime_error(
                    "Station gains are outside the row-beam cache scope");
        }
        if (oskar_telescope_noise_enabled(telescope_cpu))
        {
            throw std::runtime_error(
                    "Thermal noise must be disabled while building the "
                    "deterministic row-beam cache");
        }
        const int num_stations = oskar_telescope_num_stations(telescope_cpu);
        for (const Row& row : rows)
        {
            if (row.time_index < 0 || row.time_index >= num_times ||
                    row.antenna1 < 0 || row.antenna1 >= num_stations ||
                    row.antenna2 < 0 || row.antenna2 >= num_stations)
            {
                throw std::runtime_error("Row time or antenna is out of range");
            }
        }
        telescope_gpu = oskar_telescope_create_copy(
                telescope_cpu, OSKAR_GPU, &status);
        work = oskar_station_work_create(OSKAR_DOUBLE, OSKAR_GPU, &status);
        check_status(status, "Create GPU telescope and work buffers");

        const std::uint64_t output_bytes =
                static_cast<std::uint64_t>(args.num_rows) *
                static_cast<std::uint64_t>(args.num_sources) *
                sizeof(std::complex<float>);
        std::fstream output;
        initialise_output(output, args.output, output_bytes);
        const double dt_days = length_sec / num_times / 86400.0;

        for (std::int64_t source_first = 0;
                source_first < args.num_sources;
                source_first += args.source_chunk)
        {
            const std::int64_t count = std::min(
                    args.source_chunk, args.num_sources - source_first);
            oskar_Mem* coordinates_cpu[3] = {nullptr, nullptr, nullptr};
            oskar_Mem* coordinates_gpu[3] = {nullptr, nullptr, nullptr};
            for (int axis = 0; axis < 3; ++axis)
            {
                double* data = const_cast<double*>(
                        directions.data() +
                        static_cast<std::size_t>(
                                axis * args.num_sources + source_first));
                coordinates_cpu[axis] = oskar_mem_create_alias_from_raw(
                        data, OSKAR_DOUBLE, OSKAR_CPU,
                        static_cast<std::size_t>(count), &status);
                coordinates_gpu[axis] = oskar_mem_create_copy(
                        coordinates_cpu[axis], OSKAR_GPU, &status);
            }
            jones = oskar_jones_create(
                    OSKAR_DOUBLE_COMPLEX_MATRIX,
                    OSKAR_GPU,
                    num_stations,
                    static_cast<int>(count),
                    &status);
            check_status(status, "Allocate Jones block");

            for (int time_index = 0; time_index < num_times; ++time_index)
            {
                std::vector<std::int64_t> active_rows;
                std::set<int> station_set;
                for (std::int64_t row_index = 0;
                        row_index < args.num_rows; ++row_index)
                {
                    const Row& row = rows[static_cast<std::size_t>(row_index)];
                    if (row.time_index == time_index)
                    {
                        active_rows.push_back(row_index);
                        station_set.insert(row.antenna1);
                        station_set.insert(row.antenna2);
                    }
                }
                if (active_rows.empty()) continue;
                const double current_mjd =
                        start_mjd + dt_days * (time_index + 0.5);
                const oskar_Mem* source_coords[] = {
                    coordinates_gpu[0],
                    coordinates_gpu[1],
                    coordinates_gpu[2]
                };
                oskar_evaluate_jones_E(
                        jones,
                        OSKAR_COORDS_REL_DIR,
                        static_cast<int>(count),
                        source_coords,
                        oskar_telescope_phase_centre_longitude_rad(
                                telescope_gpu),
                        oskar_telescope_phase_centre_latitude_rad(
                                telescope_gpu),
                        telescope_gpu,
                        time_index,
                        start_mjd,
                        current_mjd,
                        frequency_hz,
                        work,
                        &status);
                check_status(status, "Evaluate Jones E");

                const std::vector<int> stations(
                        station_set.begin(), station_set.end());
                std::map<int, std::size_t> station_slot;
                for (std::size_t slot = 0; slot < stations.size(); ++slot)
                {
                    station_slot[stations[slot]] = slot;
                }
                station_cpu = oskar_mem_create(
                        OSKAR_DOUBLE_COMPLEX_MATRIX,
                        OSKAR_CPU,
                        stations.size() * static_cast<std::size_t>(count),
                        &status);
                for (std::size_t slot = 0; slot < stations.size(); ++slot)
                {
                    oskar_mem_copy_contents(
                            station_cpu,
                            oskar_jones_mem_const(jones),
                            slot * static_cast<std::size_t>(count),
                            static_cast<std::size_t>(stations[slot]) *
                                    static_cast<std::size_t>(count),
                            static_cast<std::size_t>(count),
                            &status);
                }
                check_status(status, "Copy selected station Jones matrices");
                const double4c* station_data =
                        oskar_mem_double4c_const(station_cpu, &status);
                check_status(status, "Access station Jones matrices");

                std::vector<std::complex<float>> factors(
                        static_cast<std::size_t>(count));
                for (const std::int64_t row_index : active_rows)
                {
                    const Row& row = rows[static_cast<std::size_t>(row_index)];
                    const double4c* first = station_data +
                            station_slot.at(row.antenna1) *
                                    static_cast<std::size_t>(count);
                    const double4c* second = station_data +
                            station_slot.at(row.antenna2) *
                                    static_cast<std::size_t>(count);
#pragma omp parallel for
                    for (std::int64_t source = 0; source < count; ++source)
                    {
                        factors[static_cast<std::size_t>(source)] =
                                stokes_i_factor(
                                        first[static_cast<std::size_t>(source)],
                                        second[static_cast<std::size_t>(source)]);
                    }
                    const std::uint64_t offset =
                            (static_cast<std::uint64_t>(row_index) *
                             static_cast<std::uint64_t>(args.num_sources) +
                             static_cast<std::uint64_t>(source_first)) *
                            sizeof(std::complex<float>);
                    output.seekp(static_cast<std::streamoff>(offset));
                    output.write(
                            reinterpret_cast<const char*>(factors.data()),
                            static_cast<std::streamsize>(
                                    factors.size() *
                                    sizeof(std::complex<float>)));
                    if (!output)
                    {
                        throw std::runtime_error("Failed writing beam factors");
                    }
                }
                int cleanup_status = 0;
                oskar_mem_free(station_cpu, &cleanup_status);
                station_cpu = nullptr;
            }
            output.flush();
            int cleanup_status = 0;
            oskar_jones_free(jones, &cleanup_status);
            jones = nullptr;
            for (int axis = 0; axis < 3; ++axis)
            {
                oskar_mem_free(coordinates_gpu[axis], &cleanup_status);
                oskar_mem_free(coordinates_cpu[axis], &cleanup_status);
            }
            const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
            std::cout
                    << "{\"event\":\"source_chunk\",\"source_stop\":"
                    << (source_first + count)
                    << ",\"num_sources\":" << args.num_sources
                    << ",\"elapsed_seconds\":" << elapsed << "}"
                    << std::endl;
        }
        output.close();
        const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
        std::cout
                << "{\"event\":\"complete\",\"num_rows\":" << args.num_rows
                << ",\"num_sources\":" << args.num_sources
                << ",\"frequency_hz\":" << frequency_hz
                << ",\"elapsed_seconds\":" << elapsed << "}"
                << std::endl;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << std::endl;
        status = status ? status : 1;
    }

    int cleanup_status = 0;
    oskar_mem_free(station_cpu, &cleanup_status);
    oskar_jones_free(jones, &cleanup_status);
    oskar_station_work_free(work, &cleanup_status);
    oskar_telescope_free(telescope_gpu, &cleanup_status);
    oskar_telescope_free(telescope_cpu, &cleanup_status);
    oskar_log_free(log);
    oskar::SettingsTree::free(settings);
    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
