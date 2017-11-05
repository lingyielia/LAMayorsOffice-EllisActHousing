# LAMayorsOffice-EllisActHousing
LA Mayor's Office - Ellis Act/Housing DemFree DataDive repo for all code


### Here's a description of data columns in the .csv

**Property ID**
    This is a unique property identifier that we computed (not LA). This was made by fuzzy matching on combinations of street address and ZIP code so we can track different permits coming from the same location

**APN**
    The Los Angeles County "Assessor Parcel Number". This is usually a good way to identify a property, but the value can change following big developments that split/merge properties.

**General Category**
    The general cagegory for this data record. The options are:
    * Is in RSO Inventory: This record means that the property is rent stabilized
    * Ellis Withdrawal
    * Entitlement Change
    * Building Permits
    * Demolition Permits

**Status**
    Status of a permit or of a Building/Demolition permit applications.
    There's a lot of them described in this doc [here](https://docs.google.com/document/d/1SiUQk5Vn8dibZ7ziRbMFd9l-vnKI5w4lb2J3GSlFJmE/edit)

**Status Date**
    Original issue date or application date for the permit

**Completion Date**
    Completion date for the permit actions (if applicable)

**Permit #**
    For an Entitlement Change, this is the 'Case Number'. These case numbers are described in *DCP Case Coding - {Active, Discontinued}.pdf* For building/demo permits, these are the permit numbers.

**Permit Type**
    Permit application type.

**Permit Sub-Type**
    The permit sub-type determines whether the permit application is for a 1 or 2 family dwelling, a multi-family dwelling, or a commercial structure.

**Work Description**
    Describes the work to be performed under the permit application.

**Address Full**
    String containing the address of the permit location

**Address Number**
    Address number for the permit in `Address Full`. For records where this wasn't explicitly provided, it was parsed out using the package [`usaddress`](https://pypi.python.org/pypi/usaddress).

**Address Number (float)**
    `Address number` cast to a numeric. If the string-version of the number was actually a range (i.e., 100-120), this number is the lower bound of the range.

**Street Direction**
    The cardinal direction for a street address in `Address Full`. For records where this wasn't explicitly provided, it was parsed out using the package [`usaddress`](https://pypi.python.org/pypi/usaddress).

**Street Name**
    The name of the street in `Address Full`. For records where this wasn't explicitly provided, it was parsed out using the package [`usaddress`](https://pypi.python.org/pypi/usaddress).

**Street Suffix**
    The suffix for the street in `Address Full` (e.g., St, Ave, Blvd). For records where this wasn't explicitly provided, it was parsed out using the package [`usaddress`](https://pypi.python.org/pypi/usaddress).

**City**

**State**

**Zip Code**

**Unit Count**
    Number of units in this building

**Unit Number**
    Unit number relevant for this permit

**Unit Type**

**Council District**
    The City of Los Angeles council district that has jurisdiction over the property.
