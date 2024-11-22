import knime.extension as knext
from utils import knutills as kutil
import logging
from PIL import Image, ImageOps, ImageFilter
import io
import pandas as pd

LOGGER = logging.getLogger(__name__)


# define sub-category here
knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)


@knext.node(
    name="Image Processor",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icon.png",
    category=knimeVis_category,
    id="ímg-blurrer",
)
@knext.input_table(
    name="Input Data",
    description="Table containing the image column.",
)
@knext.output_image(
    name="PNG Output Image of a plot",
    description="A port that produces a PNG image.",
)
@knext.output_image(
    name="SVG Output Image of KNIME",
    description="A port that produces an SVG image.",
)
@knext.output_table(
    name="Output table with SVG XML",
    description="A table port that contains the XML string for the KNIME logo above.",
)
class ImageNode:

    target_column = knext.ColumnParameter(
        label="Image Column",
        description="Select an Image column for blurring.",
        port_index=0,
        column_filter=kutil.is_png,
    )

    radius_box_blur = knext.IntParameter(
        label="Radius",
        description="Value between 0 to 9 to toggle size of square box used for blurring in each direction.",
        default_value=2,
        min_value=0,
        max_value=10,
    )

    def configure(
        self, config_context: knext.ConfigurationContext, input_schema_1: knext.Schema
    ):
        # pick only image columns
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]

        LOGGER.debug(f"ImageNode configure: {image_columns}")
        
        # by default select the last column
        if self.target_column is None:
            self.target_column = image_columns[-1][0]
        return (
            knext.ImagePortObjectSpec(knext.ImageFormat.PNG),
            knext.ImagePortObjectSpec(knext.ImageFormat.SVG),
            input_schema_1.append([knext.Column(knext.string(), "xml-string")]),
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):

        df = input_table.to_pandas()
        image = df[self.target_column].iloc[0]

        filtered_image = image.filter(ImageFilter.BoxBlur(radius=self.radius_box_blur))

        LOGGER.debug(f"ImageNode execute: {image}")
        buffer_png = io.BytesIO()
        filtered_image.save(buffer_png, format="png")

        svg_xml_string = """<svg xmlns="http://www.w3.org/2000/svg" width="480" height="124" viewBox="0 0 120 31"><g><g><g><path fill="#fdd800" d="M15.709 29.627l17.356-1.165.68 1.165zm-4.339-15.46l-7.658 15.46h-1.36zm15.683 3.988L17.382 3.859l.68-1.166zM30.92 26l.941 1.348L15.84 28.41zM7 29l-1.72.135 7.083-14.27zm9.284-22.06L17 5.44l9 13.208zm12.442 16.627L30 25.095 16.937 27.22zm-18.558 4.765l-1.987.337 4.679-12.276zM15.29 10L16 8.184l8.36 10.126zm11.135 11.34l1.699 1.528-9.593 2.978zm-13.174 6.138l-2.196.673 2.196-9.686zm1.202-14.348L15 10.903l7.322 6.734zm-.522 14.14V15.462l.366-1.631 10.324 5.904 1.255 1.14-10.325 5.905zM18.062 0L0 31h36.097z"/></g><g><path fill="#3e3b3a" d="M44.726 6.137c0 1.476-.863 2.408-2.248 2.408-1.385 0-2.248-.932-2.248-2.382 0-1.45.915-2.383 2.3-2.383 1.307.026 2.196.958 2.196 2.357zM55.546 5c.968 0 1.282.645 1.282 1.551v1.917H56.2V6.81c0-.647-.052-1.269-.784-1.269a.933.933 0 0 0-.81.459 1.7 1.7 0 0 0-.157.862v1.606h-.627V6.085c0-.363 0-.725-.026-.984h.627c-.026.155 0 .362 0 .544.235-.389.68-.645 1.124-.645zm-5.122 1.94c0 .595.47 1.06 1.072 1.06h.104c.34 0 .68-.05.993-.232l.026.57c-.34.13-.68.207-1.045.207-1.124 0-1.777-.673-1.777-1.787 0-1.062.653-1.758 1.62-1.758.889 0 1.464.567 1.464 1.68v.26zm1.83-.467c0-.647-.366-.958-.863-.958s-.889.363-.941.958zm-6.483-.362c0-.363 0-.725-.026-.984h.601c0 .13.026.362.026.544A1.34 1.34 0 0 1 47.55 5C48.41 5 49 5.748 49 6.758c0 1.062-.589 1.813-1.451 1.813-.47 0-.915-.233-1.15-.647 0 .13.026.647.026.828v1.114h-.628zm.628.7c0 .672.444 1.242.993 1.242.549 0 .94-.544.94-1.243 0-.751-.332-1.243-.94-1.243-.575 0-.993.544-.993 1.243zM42.478 8C43.418 8 44 7.25 44 6.163c0-1.088-.607-1.84-1.522-1.84-.941 0-1.542.752-1.542 1.814 0 1.113.6 1.863 1.542 1.863z"/></g><g><path fill="#3e3b3a" d="M68.139 5c.104 0 .209 0 .313.076v.595c-.104-.051-.235-.051-.34-.051-.575 0-.94.492-.94 1.38v1.494h-.628V6.138c0-.363 0-.726-.026-.985h.627c.026.156.053.337.053.518.156-.388.548-.671.94-.671zm-2.51 1.81c0 .96-.784 1.736-1.751 1.736a1.743 1.743 0 0 1-1.751-1.735A1.725 1.725 0 0 1 63.72 5h.157c1.02 0 1.751.775 1.751 1.81zm-2.823 0c0 .726.445 1.19 1.098 1.19C64.558 8 65 7.536 65 6.81c0-.724-.469-1.268-1.122-1.268-.653 0-1.072.518-1.072 1.269zM61.656 4h-.313c-.419 0-.575.195-.575.79v.337h.914v.493h-.914v2.848h-.654V5.62h-.784v-.493h.81v-.31c0-.817.288-1.347 1.098-1.347.157 0 .314.026.47.052z"/></g><g><path fill="#3e3b3a" d="M103.09 5c.966 0 1.28.649 1.28 1.555v1.917h-.627V6.814c0-.647-.079-1.269-.743-1.269-.381 0-.695.181-.878.455-.122.296-.183.581-.122.866v1.606h-.662V6.089c0-.363 0-.725-.026-.984h.627c0 .155 0 .362.061.544.2-.389.645-.649 1.09-.649zm-2.693 1.814c.052.907-.654 1.684-1.594 1.735h-.157a1.69 1.69 0 0 1-1.752-1.631v-.13A1.691 1.691 0 0 1 98.49 5h.157c1.02 0 1.75.778 1.75 1.814zm-2.823 0C97.574 7.54 98 8 98.646 8c.627 0 1.071-.46 1.071-1.186 0-.725-.444-1.269-1.097-1.269-.62 0-1.046.518-1.046 1.27zM96 4.51h-.726V3.81H96zm0 3.963h-.7V5.105h.7zM94.464 5.13v.492h-.915v1.476c0 .544.078.901.575.901.13 0 .261 0 .366-.072l.026.492c-.183.052-.366.103-.516.103-.896 0-1.079-.523-1.079-1.243V5.623h-.784V5.13h.784v-.803l.628-.207v1.01zM89 5.312A2.258 2.258 0 0 1 90.099 5c.993 0 1.359.597 1.359 1.374v1.088c0 .362 0 .777.026 1.01h-.575v-.285-.311a1.195 1.195 0 0 1-1.124.622c-.732 0-1.176-.389-1.176-1.01 0-.7.653-1.088 1.777-1.088.13 0 .314 0 .47.026V6.4c0-.57-.287-.855-.856-.855-.372 0-.712.13-1 .311zm.288 2.201c0 .337.262.487.654.487.366 0 .68-.15.836-.46.079-.208.13-.441.105-.674h-.34c-.837 0-1.255.207-1.255.647zm-2.901-.336c.078.207.183.57.235.823.052-.253.183-.642.262-.823l.68-2.046h.653L87 8.498h-.796L85 5.13h.655zm-1.777-.363c.052.907-.68 1.71-1.61 1.735h-.142a1.69 1.69 0 0 1-1.751-1.631v-.104A1.73 1.73 0 0 1 82.701 5h.157c1.046 0 1.752.778 1.752 1.814zm-2.797 0c0 .725.444 1.186 1.071 1.186C83.512 8 84 7.54 84 6.814c0-.725-.488-1.269-1.142-1.269-.653 0-1.045.518-1.045 1.27zM78.938 5c.967 0 1.254.649 1.254 1.555v1.917h-.627V6.814c0-.647-.052-1.269-.784-1.269-.34 0-.654.181-.781.455-.16.296-.213.581-.186.866v1.606h-.628V6.089c0-.363 0-.725-.026-.984h.627c0 .155 0 .362.027.544C78 5.26 78.494 5 78.938 5zm-4.183 0C75.723 5 76 5.649 76 6.555v1.917h-.591V6.814c0-.647-.052-1.269-.784-1.269-.34 0-.625.181-.81.455-.131.296-.183.581-.157.866v1.606H73V5.105h.632c-.027.155 0 .362 0 .544.235-.389.68-.649 1.123-.649zM71.75 8.472h-.68V3.888h.68z"/></g><g><path fill="#3e3b3a" d="M43.706 13.21v8.21h.079l6.587-8.21h3.946l-6.69 8.365L56.277 31h-4.443l-8.129-8.752V31H40.23V13.21zm17.294 0l10.7 12.405V13.21h3.398V31h-2.98L61.402 18.364V31H58V13.21zm21.47 0V31H79V13.21zm7.004 0l5.907 6.992 5.881-6.992h3.085V31h-3.45V18.493h-.105l-5.175 6.268h-.497l-5.201-6.268h-.105V31h-3.476V13.21zm30.346 0v2.693h-8.128v4.662h7.789v2.693h-7.79v5.05H120V31h-11.785V13.21z"/></g></g></g></svg>"""
        buffer_svg = io.BytesIO()
        buffer_svg.write(svg_xml_string.encode("utf-8"))

        df["xml-string"] = svg_xml_string

        return (
            buffer_png.getvalue(),
            buffer_svg.getvalue(),
            input_table.from_pandas(df),
        )
