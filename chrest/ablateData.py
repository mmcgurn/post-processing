import argparse
import pathlib
import sys

import numpy as np
import h5py
import pandas as pd

from chrestData import ChrestData
from supportPaths import expand_path
from scipy.spatial import KDTree


class AblateData:
    """
    Creates a new class from hdf5 chrest formatted file(s).
    """

    def __init__(self, files=None):
        if files is None:
            self.files = []
        elif isinstance(files, str):
            self.files = expand_path(files)
        elif isinstance(files, pathlib.Path):
            self.files = expand_path(files)
        else:
            self.files = files

        # store the files based upon time
        self.files_per_time = dict()
        self.times = []

        # store the cells and vertices
        self.cells = None
        self.vertices = None

        # load the metadata from the first file
        self.metadata = dict()

        # Open each file to get the time and check the available fields
        for file in self.files:
            # Load in the hdf5 file
            hdf5 = h5py.File(file, 'r')

            # If not set, copy over the cells and vertices
            if self.cells is None:
                self.cells = hdf5["viz"]["topology"]["cells"]
                self.vertices = hdf5["geometry"]["vertices"][:]

            # extract the time
            time = hdf5['time'][0][0]
            self.times.append(time)

            self.files_per_time[time] = file

        # Store the list of times
        self.times = list(self.files_per_time.keys())
        self.times.sort()

        # Load in any available metadata based upon the file structure
        self.metadata['path'] = str(self.files[0])

        # load in the restart file
        try:
            restart_file = self.files[0].parent.parent / "restart.rst"
            if restart_file.is_file():
                self.metadata['restart.rst'] = restart_file.read_text()
        except (Exception,):
            print("Could not locate restart.rst for metadata")

        # load in the restart file
        try:
            simulation_directory = self.files[0].parent.parent
            yaml_files = list(simulation_directory.glob("*.yaml"))
            for yaml_file in yaml_files:
                self.metadata[yaml_file.name] = yaml_file.read_text()
        except (Exception,):
            print("Could not locate yaml files for metadata")

    """
    Reports the fields from each the file
    """

    def get_fields(self):
        fields_names = []

        if len(self.files) > 0:
            with h5py.File(self.files[0], 'r') as hdf5:
                fields_names.extend(hdf5['cell_fields'].keys())

        return fields_names

    """
    computes the cell center for each cell [c, d]
    """

    def compute_cell_centers(self, dimensions):
        # create a new np array based upon the dim
        number_cells = self.cells.shape[0]

        vertices = self.vertices[:]
        vertices_dim = vertices.shape[1]

        coords = np.zeros((number_cells, dimensions))

        # march over each cell
        for c in range(len(coords)):
            cell_vertices = vertices.take(self.cells[c], axis=0)

            # take the average
            cell_center = np.sum(cell_vertices, axis=0)
            cell_center = cell_center / len(cell_vertices)

            # put back
            coords[c, 0:vertices_dim] = cell_center

        return coords

    """
    gets the specified field and the number of components
    """

    def get_field(self, field_name, component_names=None):
        # create a dictionary of times/data
        data = []
        components = 0
        all_component_names = None
        # march over the files
        for t in self.times:
            # Load in the file
            with h5py.File(self.files_per_time[t], 'r') as hdf5:
                try:
                    # Check each of the cell fields
                    hdf5_field = hdf5['cell_fields'][field_name]

                    if len(hdf5_field[0, :].shape) > 1:
                        components = hdf5_field[0, :].shape[1]
                        # check for component names
                        all_component_names = [""] * components

                        for i in range(components):
                            if ('componentName' + str(i)) in hdf5_field.attrs:
                                all_component_names[i] = hdf5_field.attrs['componentName' + str(i)].decode("utf-8")

                    # load in the specified field name, but check if we should down select components
                    if component_names is None:
                        hdf_field_data = hdf5_field[0, :]
                        data.append(hdf_field_data)
                    else:
                        # get the indexes
                        indexes = []
                        for component_name in component_names:
                            try:
                                indexes.append(all_component_names.index(component_name))
                            except ValueError as error:
                                raise Exception("Cannot locate " + component_name + " in field " + field_name + ". ",
                                                error)

                        # sort the index list in order
                        index_name_list = zip(indexes, component_names)
                        index_name_list = sorted(index_name_list)
                        indexes, component_names = zip(*index_name_list)

                        # down select only the components that were asked for
                        hdf_field_data = hdf5_field[0, ..., list(indexes)]
                        data.append(hdf_field_data)
                        all_component_names = list(component_names)
                        components = len(all_component_names)
                except Exception as e:
                    raise Exception(
                        "Unable to open field " + field_name + "." + str(e))

        return np.stack(data), components, all_component_names

    """
    converts the supplied fields ablate object to chrest data object.
    """

    def map_to_chrest_data(self, chrest_data, field_mapping, field_select_components=dict(),
                           max_distance=sys.float_info.max):
        # get the cell centers for this mesh
        cell_centers = self.compute_cell_centers(chrest_data.dimensions)

        # add the ablate times to chrest
        chrest_data.times = self.times.copy()

        # store the fields in ablate we need
        ablate_fields = list(field_mapping.keys())

        # store the chrest data in the same order
        ablate_field_data = []
        chrest_field_data = []
        components = []

        # create the new field in the chrest data
        for ablate_field in ablate_fields:
            chrest_field = field_mapping[ablate_field]

            # check to see if there is a down selection of components
            component_select_names = field_select_components.get(ablate_field)

            # get the field from ablate
            ablate_field_data_tmp, components_tmp, component_names = self.get_field(ablate_field,
                                                                                    component_select_names)

            ablate_field_data.append(ablate_field_data_tmp)
            components.append(components_tmp)
            chrest_field_data.append(chrest_data.create_field(chrest_field, components_tmp, component_names)[0])

        # build a list of k, j, i points to iterate over
        chrest_coords = chrest_data.get_coordinates()

        # size up the storage value
        chrest_cell_number = np.prod(chrest_data.grid)

        # reshape to get a single list order back
        chrest_cell_centers = chrest_coords.reshape((chrest_cell_number, chrest_data.dimensions))

        # now search and copy over data
        tree = KDTree(cell_centers)
        dist, points = tree.query(chrest_cell_centers)

        # march over each field
        for f in range(len(ablate_field_data)):
            # get in the correct order
            ablate_field_in_chrest_order = ablate_field_data[f][:, points]

            setToZero = np.where(dist > max_distance)

            ablate_field_in_chrest_order[:, setToZero] = 0.0

            # reshape it back to k,j,i
            ablate_field_in_chrest_order = ablate_field_in_chrest_order.reshape(
                chrest_field_data[f].shape
            )

            # copy over the data
            chrest_field_data[f][:] = ablate_field_in_chrest_order[:]

        # copy over the metadata
        chrest_data.metadata = self.metadata


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a chrest data file from an ablate file')
    parser.add_argument('--file', dest='file', type=pathlib.Path, required=True,
                        help='The path to the ablate hdf5 file(s) containing the ablate data.'
                             '  A wild card can be used '
                             'to supply more than one file.')
    parser.add_argument('--start', dest='start', type=float,
                        help='Optional starting point for chrest data. Default is [0., 0.0, -0.0127]',
                        nargs='+', default=[0., 0.0, -0.0127]
                        )

    parser.add_argument('--end', dest='end', type=float,
                        help='Optional ending point for chrest data. Default is [0.1, 0.0254, 0.0127]',
                        nargs='+', default=[0.1, 0.0254, 0.0127]
                        )

    parser.add_argument('--delta', dest='delta', type=float,
                        help='Optional grid spacing for chrest data. Default is [0.0005, 0.0005, 0.0005]',
                        nargs='+', default=[0.0005, 0.0005, 0.0005]
                        )

    parser.add_argument('--fields', dest='fields', type=str,
                        help='The list of fields to map from ablate to chrest in format  --field '
                             'ablate_name:chrest_name:components --field aux_temperature:temperature '
                             'aux_velocity:vel. Components are optional list such as H2,N2',
                        nargs='+', default=["aux_temperature:temperature", "aux_velocity:vel"]
                        )

    parser.add_argument('--print_fields', dest='print_fields', action='store_true',
                        help='If true, prints the fields available to convert', default=False
                        )

    parser.add_argument('--max_distance', dest='max_distance', type=float,
                        help="The max distance to search for a point in ablate", default=sys.float_info.max)

    args = parser.parse_args()

    files = expand_path(args.file)

    result_dir = files[0].parent / "csv"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Open each file to get the time and check the available fields
    for file in files:
        ablate_data = AblateData(file)

        # create a chrest data
        chrest_data = ChrestData()
        chrest_data.setup_new_grid(args.start, args.end, args.delta)

        # map the ablate data to chrest
        field_mapping = {"solution_euler": "euler"}
        component_mapping = {"solution_euler": ["rho", "rhoVel0", "rhoVel1", "rhoVel2"]}
        ablate_data.map_to_chrest_data(chrest_data, field_mapping, component_mapping)

        linspaces = []
        for dim in range(chrest_data.dimensions):
            linspaces.append(
                np.linspace(chrest_data.start_point[dim] + chrest_data.delta[dim] / 2.0,
                            chrest_data.end_point[dim] - chrest_data.delta[dim] / 2.0,
                            chrest_data.grid[dim], endpoint=True))

        # Save the result data
        data = chrest_data.get_field("euler")[0][0, :, 0, :, 1]
        frame = pd.DataFrame(data, columns=linspaces[0], index=linspaces[2])
        frame.to_csv(result_dir / f'{file.stem}.u.csv')

        data = chrest_data.get_field("euler")[0][0, :, 0, :, 2]
        frame = pd.DataFrame(data, columns=linspaces[0], index=linspaces[2])
        frame.to_csv(result_dir / f'{file.stem}.v.csv')

        data = chrest_data.get_field("euler")[0][0, :, 0, :, 3]
        frame = pd.DataFrame(data, columns=linspaces[0], index=linspaces[2])
        frame.to_csv(result_dir / f'{file.stem}.w.csv')
        #
        #
        # dataU = pd.read_csv("/Users/mcgurn/Downloads/foc-open-700/foo.u.csv")
        # dataW = pd.read_csv("/Users/mcgurn/Downloads/foc-open-700/foo.w.csv")

        # import matplotlib.pyplot as plt
        #
        # fig, ax = plt.subplots()
        #
        # x = dataW.columns[1::10].astype(float)
        # y = dataW.iloc[::10, 0]
        # dataU = dataU.iloc[::10, 1::10]
        # dataW = dataW.iloc[::10, 1::10]
        # #
        # # ax.contourf(x, y, dataW)
        # # ax.set_xlabel('x')
        # # ax.set_ylabel('y')
        # # ax.set_aspect('equal')
        #
        # ax.quiver(x, y,  dataU, dataW)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_aspect('equal')
        # fig.show()
